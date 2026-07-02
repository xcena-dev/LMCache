# SPDX-License-Identifier: Apache-2.0
"""Maru-backed L1 manager (sibling of :class:`L1Manager`).

``MaruL1Manager`` is the maru counterpart to the default
:class:`~lmcache.v1.distributed.l1_manager.L1Manager`. Both expose the
same public method surface so ``StorageManager`` can hold either behind
a single field, selected by one branch at construction time:

    self._l1_manager = (
        MaruL1Manager(cfg)
        if cfg.memory_config.maru_config is not None
        else L1Manager(cfg)
    )

Design (see ``docs/design/v1/distributed/maru_l1_l2_impl_design_kr.md``):

- ``MaruL1Manager`` **owns** a :class:`MaruMemoryAllocator` (built from
  ``config.memory_config.maru_config``) directly, rather than routing
  through the shared :class:`L1MemoryManager`. This keeps the default
  ``L1Manager`` / ``L1MemoryManager`` free of maru branches (they stay
  byte-identical to ``dev``).
- The physical store / retrieve / delete RPCs are delegated to an
  internal :class:`MaruL1Dispatcher`, reusing its existing, tested
  logic (side channels, prefix-pin semantics, batch_store dup-skip).

PR1 scope (parity with the current temporary integration): maru runs
**L1-only** with the L2 controller stack bypassed at ``StorageManager``.
The state-machine methods that ``L1Manager`` implements over its object
dict (``touch_keys`` / ``is_key_evictable`` / ``get_object_state`` /
``memcheck``) are provided here at parity — MaruServer owns eviction
decisions and there is no in-process object dict — and listener firing /
L2 tiering land in later PRs.
"""

# Future
from __future__ import annotations

# Standard
from typing import Literal

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1MemoryDesc,
    L1OperationResult,
)
from lmcache.v1.distributed.l1_protocol import MaruL1ManagerInterface
from lmcache.v1.distributed.maru_l1_dispatch import MaruL1Dispatcher
from lmcache.v1.distributed.maru_memory_allocator import (
    MaruL1Config,
    MaruMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


class _MaruAllocatorMemoryManager:
    """Thin adapter presenting the :class:`MaruL1Dispatcher`-facing slice
    of the ``L1MemoryManager`` API on top of a raw
    :class:`MaruMemoryAllocator`.

    ``MaruL1Dispatcher`` was written against the shared
    ``L1MemoryManager`` and uses exactly two of its methods —
    :meth:`allocate` (in ``reserve_write``) and :meth:`get_memory_usage`
    (in ``report_status``). Rather than pull the maru allocator back
    through ``L1MemoryManager`` (which the sibling design deliberately
    keeps maru-free), :class:`MaruL1Manager` owns this small adapter and
    hands it to the dispatcher.

    It also exposes :attr:`allocator` and :meth:`register_kv_layout` for
    :class:`MaruL1Manager` itself.
    """

    def __init__(self, allocator: MaruMemoryAllocator) -> None:
        self._allocator = allocator

    @property
    def allocator(self) -> MaruMemoryAllocator:
        """The owned :class:`MaruMemoryAllocator`."""
        return self._allocator

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate ``count`` CXL-backed memory objects.

        Args:
            layout_desc: Description of the memory layout to allocate.
            count: Number of memory objects to allocate.

        Returns:
            ``(L1Error.SUCCESS, objects)`` on success, or
            ``(L1Error.OUT_OF_MEMORY, [])`` if the pool cannot satisfy
            the request.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def get_memory_usage(self) -> tuple[int, int]:
        """Best-effort CXL pool usage via ``MaruHandler.get_stats``.

        Eviction is owned by MaruServer, so this is observability only.
        Returns ``(0, 0)`` before ``init_layout`` (handler not yet
        built) and on any handler error rather than crashing callers.

        Returns:
            ``(used_bytes, total_bytes)``.
        """
        if not self._allocator.is_initialized:
            return 0, 0
        try:
            handler = self._allocator.handler
            stats = handler.get_stats() if hasattr(handler, "get_stats") else {}
            used = int(stats.get("used_bytes", 0))
            total = int(stats.get("pool_size_bytes", 0) or stats.get("pool_size", 0))
            return used, total
        except Exception:
            logger.exception("Failed to query Maru handler stats")
            return 0, 0

    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout to the maru allocator.

        Forwards to :meth:`MaruMemoryAllocator.init_layout`, which brings
        the CXL pool up on the first call and validates layout
        consistency (single-model constraint) on subsequent calls.

        Args:
            shapes: KV chunk shapes (per-layer-group).
            dtypes: KV chunk dtypes aligned with ``shapes``.
            fmt: Memory format.
            chunk_size_in_tokens: LMCache chunk size in tokens.
        """
        self._allocator.init_layout(shapes, dtypes, fmt, chunk_size_in_tokens)

    def close(self) -> None:
        """Close the underlying allocator (CXL adapter + handler)."""
        self._allocator.close()


class MaruL1Manager(MaruL1ManagerInterface):
    """Maru-backend L1 manager — a sibling of :class:`L1Manager`.

    Implements the full ``L1Manager`` public surface. RPC-driven
    operations (``reserve_read`` / ``unsafe_read`` / ``finish_read`` /
    ``reserve_write`` / ``finish_write`` / ``finish_write_and_reserve_read``
    / ``delete`` / ``clear`` / ``report_status``) are delegated to an
    internal :class:`MaruL1Dispatcher`. The remaining methods are
    implemented directly at PR1 parity (see module docstring).

    In maru mode LMCache does not keep an in-process object dict / TTL
    state machine — MaruServer owns the shared KV index and pin counts —
    so the state-machine query methods return simple parity answers.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        maru_config = config.memory_config.maru_config
        if maru_config is None:
            raise ValueError(
                "MaruL1Manager requires config.memory_config.maru_config to be set"
            )
        self._config: MaruL1Config = maru_config
        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        # MaruL1Manager owns its allocator directly (sibling design):
        # the shared L1MemoryManager is deliberately kept maru-free.
        self._allocator = MaruMemoryAllocator(maru_config)
        self._memory_manager = _MaruAllocatorMemoryManager(self._allocator)

        # Physical store/retrieve/delete RPC logic lives in the dispatcher
        # (reused unchanged). It drives allocation through
        # ``_memory_manager`` and reads usage stats through it.
        self._dispatcher = MaruL1Dispatcher(
            allocator=self._allocator,
            memory_manager=self._memory_manager,  # type: ignore[arg-type]
            write_ttl_seconds=self._write_ttl_seconds,
            read_ttl_seconds=self._read_ttl_seconds,
        )

        # Accepted-and-stored for API parity; listener firing is a later
        # PR (PR3, when the L2 controller stack is wired for maru).
        self._registered_listeners: list[L1ManagerListener] = []

    # ------------------------------------------------------------------
    # Layout binding (maru-only; forwarded from StorageManager)
    # ------------------------------------------------------------------

    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout, bringing up the CXL pool on first call.

        Forwarded from ``StorageManager.register_kv_layout``, which is in
        turn invoked once a vLLM worker exposes its KV cache tensors.

        Args:
            shapes: KV chunk shapes (per-layer-group).
            dtypes: KV chunk dtypes aligned with ``shapes``.
            fmt: Memory format.
            chunk_size_in_tokens: LMCache chunk size in tokens.
        """
        self._memory_manager.register_kv_layout(
            shapes, dtypes, fmt, chunk_size_in_tokens
        )

    # ------------------------------------------------------------------
    # Listener registration (parity: accept & store; firing is PR3)
    # ------------------------------------------------------------------

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener.

        Stored for API parity with :class:`L1Manager`. In PR1 maru
        bypasses the StoreController / PrefetchController / eviction
        stack, so these listeners are never fired; wiring them lands
        with the L2 tiering PR.

        Args:
            listener: The listener to register.
        """
        self._registered_listeners.append(listener)

    # ------------------------------------------------------------------
    # Read path (delegated to the dispatcher)
    # ------------------------------------------------------------------

    def reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Pin + retrieve the given keys, staging their MemoryObjs.

        Delegates to :meth:`MaruL1Dispatcher.reserve_read`. ``extra_count``
        is accepted for signature parity with :class:`L1Manager` but is
        not used by the maru flow (no per-key TTL read-lock counting).

        Args:
            keys: Object keys to reserve read access for.
            extra_count: Unused in maru mode (accepted for parity).

        Returns:
            A dict mapping each key to ``(L1Error, MemoryObj | None)``.
        """
        del extra_count  # unused in maru mode
        return self._dispatcher.reserve_read(keys)

    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Return MemoryObjs staged by :meth:`reserve_read`.

        Args:
            keys: Object keys to read.

        Returns:
            A dict mapping each key to ``(L1Error, MemoryObj | None)``.
        """
        return self._dispatcher.unsafe_read(keys)

    def finish_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1Error]:
        """Drop staged read entries and unpin the keys on MaruServer.

        Args:
            keys: Object keys to finish read access for.
            extra_count: Unused in maru mode (accepted for parity).

        Returns:
            A dict mapping each key to an :class:`L1Error`.
        """
        del extra_count  # unused in maru mode
        return self._dispatcher.finish_read(keys)

    # ------------------------------------------------------------------
    # Write path (delegated to the dispatcher)
    # ------------------------------------------------------------------

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Allocate CXL MemoryObjs for the given keys.

        Args:
            keys: Object keys to reserve write access for.
            is_temporary: Per-key temporary flags (unused in maru mode).
            layout_desc: Memory layout for the objects to allocate.
            mode: Reservation mode (unused in maru mode; the maru flow
                only uses ``"new"``).

        Returns:
            A dict mapping each key to ``(L1Error, MemoryObj | None)``.
        """
        return self._dispatcher.reserve_write(keys, is_temporary, layout_desc, mode)

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Register reserved KVs with MaruServer via ``batch_store``.

        Args:
            keys: Object keys to finish write access for.

        Returns:
            A dict mapping each key to an :class:`L1Error`.
        """
        return self._dispatcher.finish_write(keys)

    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomic write-to-read transition (defensive in maru mode).

        The maru flow stages MemoryObjs in the read side channel during
        :meth:`reserve_read` and never exercises this path, so it returns
        a safe answer for any defensive caller.

        Args:
            keys: Object keys to transition.
            extra_count: Unused in maru mode (accepted for parity).

        Returns:
            A dict mapping each key to ``(L1Error, MemoryObj | None)``.
        """
        del extra_count  # unused in maru mode
        return self._dispatcher.finish_write_and_reserve_read(keys)

    # ------------------------------------------------------------------
    # Lifecycle (delegated / parity)
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from the shared index via MaruServer.

        Args:
            keys: Object keys to delete.

        Returns:
            A dict mapping each key to an :class:`L1Error`.
        """
        return self._dispatcher.delete(keys)

    def clear(self, force: bool = False) -> None:
        """Drop staged side-channel entries.

        The CXL pool is owned by MaruServer and is never wiped by the L1
        layer; ``force`` only affects in-process read/write bookkeeping.

        Args:
            force: If True, log the count of dropped pending objects.
        """
        self._dispatcher.clear(force)

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """No-op in maru mode.

        MaruServer owns eviction decisions, so there is no in-process LRU
        bookkeeping to update and no listener is fired at PR1 parity.

        Args:
            keys: Object keys that were accessed.
        """
        del keys  # no observable effect in maru mode

    # ------------------------------------------------------------------
    # Observability / introspection (parity)
    # ------------------------------------------------------------------

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Whether a key is eligible for eviction.

        No ``L1EvictionController`` runs against maru in PR1, so this is
        never consulted on the hot path. Returns ``True`` to keep the
        contract simple for any defensive caller — MaruServer's
        ``pin_kv`` / ``delete_kv`` make the authoritative atomic decision.

        Args:
            key: The object key to check.

        Returns:
            Always ``True``.
        """
        del key
        return True

    def get_memory_usage(self) -> tuple[int, int]:
        """Best-effort CXL pool usage.

        Returns:
            ``(used_bytes, total_bytes)`` from ``MaruHandler.get_stats``;
            ``(0, 0)`` before the pool is initialized or on error.
        """
        return self._memory_manager.get_memory_usage()

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Descriptor of the L1 memory buffer, or ``None`` for maru.

        Maru-backed L1 lives in CXL pages mmap'd per region, not a single
        contiguous buffer, so there is no single ``(ptr, size, align)`` to
        describe. Returning ``None`` is the copy-type-L2 contract (the L2
        adapter factory accepts ``Optional[L1MemoryDesc]``); per-region
        descriptors for registration-type L2 are a later PR.

        Returns:
            Always ``None`` in PR1.
        """
        return None

    def get_object_state(self, key: ObjectKey) -> None:
        """Internal object state — always ``None`` in maru mode.

        There is no in-process object dict / TTLLock / ``L1ObjectState``
        in maru mode (state lives in the shared MaruServer index).

        Args:
            key: The object key.

        Returns:
            Always ``None``.
        """
        del key
        return None

    def report_status(self) -> dict:
        """Return a maru-flavoured status snapshot.

        Returns:
            A status dict (see :meth:`MaruL1Dispatcher.report_status`).
        """
        return self._dispatcher.report_status()

    def memcheck(self) -> bool:
        """Memory check — trivially healthy in maru mode.

        There is no in-process object dict to introspect; MaruServer and
        the handler own consistency.

        Returns:
            Always ``True``.
        """
        return True

    def close(self) -> None:
        """Drop pending read/write handles and close the allocator.

        The CXL page lifecycle remains owned by MaruServer; closing here
        only tears down this instance's handler connection and clears the
        in-process side channels.
        """
        self._dispatcher.clear(force=False)
        self._memory_manager.close()
