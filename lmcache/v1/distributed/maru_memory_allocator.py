# SPDX-License-Identifier: Apache-2.0

"""Maru-backed L1 memory allocator for MP mode.

This module exposes :class:`MaruMemoryAllocator`, an implementation of
:class:`MemoryAllocatorInterface` whose ``MemoryObj`` instances are
backed by CXL shared memory via the embedded ``CxlMemoryAdapter``
(``maru_lmcache``).

Lifecycle:
    The allocator is constructed eagerly (when ``L1MemoryManager`` is
    built) but the underlying ``MaruHandler`` connection and
    ``CxlMemoryAdapter`` pool are deferred until :meth:`init_layout` is
    called with the KV layout learned from the first
    ``register_kv_cache`` RPC. This matches LMCache MP's two-phase
    startup: the storage manager exists before any vLLM worker has
    registered its KV cache tensors.

Key invariants:
    - ``MemoryObj.parent_allocator`` is ``None`` for all objects
      returned by this allocator. LMCache's refcount-driven free path
      must NOT release the underlying CXL pages — lifecycle is owned
      by ``MaruServer`` (``pin_kv`` / ``unpin_kv`` / ``delete_kv``).
    - :meth:`get_by_location` and :meth:`create_store_handle` are not
      part of ``MemoryAllocatorInterface``; ``L1Manager``'s maru
      branch reaches them through an ``isinstance`` check.
    - ``maru`` and ``maru_lmcache`` are imported lazily so loading
      this module does not require those packages to be installed.

Known limitations:
    *Single-model per LMCache instance.* The CXL pool is typed at the
    first :meth:`init_layout` call (``CxlMemoryAdapter`` pre-creates
    one ``MemoryObj`` per page with the canonical
    shapes/dtypes/fmt). Subsequent registrations with a different
    layout are rejected. The default DRAM allocators
    (:class:`LazyMemoryAllocator` / :class:`MixedMemoryAllocator`)
    support multi-model deployments transparently; maru does not.
    TODO(maru-multi-model): partition the pool by layout key and hold
    one ``CxlMemoryAdapter`` per distinct layout.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, List, Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)


@dataclass
class MaruL1Config:
    """Configuration for :class:`MaruMemoryAllocator`.

    Carries only the layout-independent parameters known at storage
    manager construction time. The KV layout
    (shapes/dtypes/fmt/chunk_size) is supplied later via
    :meth:`MaruMemoryAllocator.init_layout` once a vLLM worker has
    registered its KV cache.

    Attributes:
        server_url: MaruServer endpoint. Accepts both
            ``maru://host:port`` and ``tcp://host:port``; the former
            is rewritten to the latter internally.
        pool_size_bytes: Per-instance CXL pool quota requested from
            ``MaruServer``.
        instance_id: Stable identifier for this client instance, used
            by ``MaruServer`` for ownership tracking, restart
            recovery, and observability. If ``None``, ``MaruConfig``
            auto-generates a UUID (acceptable for single-instance /
            single-node setups but not recommended for multi-node
            deployments).
        timeout_ms: Socket timeout for RPC calls.
        use_async_rpc: Whether to use async DEALER-ROUTER RPC client.
        max_inflight: Max concurrent in-flight async requests.
        eager_map: Pre-map all shared regions on connect.
    """

    server_url: str
    pool_size_bytes: int
    instance_id: Optional[str] = None
    timeout_ms: int = 5000
    use_async_rpc: bool = True
    max_inflight: int = 64
    eager_map: bool = True


class MaruMemoryAllocator(MemoryAllocatorInterface):
    """L1 memory allocator backed by CXL shared memory via Maru.

    Wraps the embedded ``CxlMemoryAdapter`` (``maru_lmcache``) and
    re-exposes the slice of its API required by
    :class:`MemoryAllocatorInterface`. Adds :meth:`get_by_location`
    and :meth:`create_store_handle` for callers — chiefly
    ``L1Manager``'s maru branch — that need to drive ``MaruHandler``
    RPCs directly.

    Lifecycle (lazy):
        ``__init__`` stores the config only — no MaruServer RPC. The
        ``MaruHandler`` connection and ``CxlMemoryAdapter`` pool are
        built on the first :meth:`init_layout` call (triggered by the
        first ``register_kv_cache`` from a vLLM worker). Calls to
        :meth:`batched_allocate` / :meth:`allocate` /
        :meth:`get_by_location` / :meth:`create_store_handle` before
        :meth:`init_layout` raise ``RuntimeError``.

    Known limitations:
        Single-model per instance — see module docstring.
        ``TODO(maru-multi-model)``.
    """

    def __init__(self, config: MaruL1Config) -> None:
        self._config = config
        self._handler: Optional[Any] = None
        self._cxl_adapter: Optional[Any] = None
        # Set in ``init_layout``; ``0`` is a sentinel for "not yet
        # initialized" and is never returned to callers (the property
        # raises before they can observe it).
        self._single_token_size: int = 0
        self._shapes: Optional[List[torch.Size]] = None
        self._dtypes: Optional[List[torch.dtype]] = None
        self._fmt: Optional[MemoryFormat] = None
        self._chunk_size_in_tokens: int = 0

    # ------------------------------------------------------------------
    # Two-phase initialization
    # ------------------------------------------------------------------

    def init_layout(
        self,
        shapes: List[torch.Size],
        dtypes: List[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind a KV layout and bring up the CXL pool.

        First call connects ``MaruHandler`` (sized to the layout's
        full-chunk byte budget) and constructs the
        ``CxlMemoryAdapter`` pool. Subsequent calls with the same
        layout are no-ops; mismatched layouts raise ``ValueError``
        (single-model constraint — see class docstring).

        Args:
            shapes: KV chunk shapes (per-layer-group when
                heterogeneous, otherwise single-element).
            dtypes: KV chunk dtypes aligned with ``shapes``.
            fmt: Memory format (e.g. ``KV_2LTD`` or ``KV_MLA_FMT``).
            chunk_size_in_tokens: LMCache chunk size in tokens
                (typically 256).

        Raises:
            ValueError: If a layout has already been bound and the new
                layout differs — maru is single-model only.
            RuntimeError: If ``MaruHandler.connect()`` fails.
        """
        if chunk_size_in_tokens <= 0:
            raise ValueError(
                f"chunk_size_in_tokens must be positive, got {chunk_size_in_tokens}"
            )

        full_chunk_size_bytes = _compute_full_chunk_size_bytes(shapes, dtypes)
        if full_chunk_size_bytes <= 0:
            raise ValueError(
                f"full_chunk_size_bytes computed to non-positive value "
                f"({full_chunk_size_bytes}) from shapes={shapes} dtypes={dtypes}"
            )
        if full_chunk_size_bytes % chunk_size_in_tokens != 0:
            raise ValueError(
                f"full_chunk_size_bytes ({full_chunk_size_bytes}) must be a "
                f"multiple of chunk_size_in_tokens ({chunk_size_in_tokens})"
            )

        if self._cxl_adapter is not None:
            if (
                self._shapes != shapes
                or self._dtypes != dtypes
                or self._fmt != fmt
                or self._chunk_size_in_tokens != chunk_size_in_tokens
            ):
                raise ValueError(
                    "MaruMemoryAllocator: layout mismatch on subsequent "
                    "register_kv_layout call. The maru backend is "
                    "single-model only — see class docstring "
                    "(TODO maru-multi-model).\n"
                    f"  existing: shapes={self._shapes} dtypes={self._dtypes} "
                    f"fmt={self._fmt} chunk={self._chunk_size_in_tokens}\n"
                    f"  new:      shapes={shapes} dtypes={dtypes} "
                    f"fmt={fmt} chunk={chunk_size_in_tokens}"
                )
            return

        # Lazy import: maru runtime is only required once a layout is
        # actually bound. Importing here keeps the module loadable on
        # non-maru deployments.
        # Third Party
        from maru import MaruConfig, MaruHandler
        from maru_lmcache import CxlMemoryAdapter

        # ``MaruHandler`` expects the ``tcp://`` scheme; ``maru://``
        # is the LMCache-facing convention (mirrors ``MaruBackend``).
        server_url = self._config.server_url
        if server_url.startswith("maru://"):
            server_url = "tcp://" + server_url[len("maru://") :]

        maru_config = MaruConfig(
            server_url=server_url,
            instance_id=self._config.instance_id,
            pool_size=self._config.pool_size_bytes,
            chunk_size_bytes=full_chunk_size_bytes,
            auto_connect=False,
            timeout_ms=self._config.timeout_ms,
            use_async_rpc=self._config.use_async_rpc,
            max_inflight=self._config.max_inflight,
            eager_map=self._config.eager_map,
        )

        handler = MaruHandler(maru_config)
        if not handler.connect():
            raise RuntimeError(
                f"Failed to connect MaruHandler to {self._config.server_url}"
            )
        logger.info(
            "[MaruMemoryAllocator] connected: server=%s instance_id=%s "
            "pool_size=%d chunk_size_bytes=%d",
            self._config.server_url,
            handler.instance_id,
            self._config.pool_size_bytes,
            full_chunk_size_bytes,
        )

        self._handler = handler
        self._cxl_adapter = CxlMemoryAdapter(
            handler=handler,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
            chunk_size=handler.get_chunk_size(),
        )
        self._shapes = shapes
        self._dtypes = dtypes
        self._fmt = fmt
        self._chunk_size_in_tokens = chunk_size_in_tokens
        self._single_token_size = full_chunk_size_bytes // chunk_size_in_tokens

    @property
    def is_initialized(self) -> bool:
        """``True`` once :meth:`init_layout` has constructed the pool."""
        return self._cxl_adapter is not None

    # ------------------------------------------------------------------
    # Accessors used by L1Manager's maru branch (via isinstance check)
    # ------------------------------------------------------------------

    @property
    def handler(self) -> Any:
        """The connected ``MaruHandler``.

        ``L1Manager``'s maru branch uses this to issue
        ``batch_store`` / ``batch_pin`` / ``batch_retrieve`` /
        ``batch_unpin`` / ``delete`` directly, bypassing the
        ``L2AdapterInterface`` framework.

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        if self._handler is None:
            raise RuntimeError(
                "MaruMemoryAllocator.handler accessed before init_layout(); "
                "the MaruHandler is built lazily on the first "
                "register_kv_cache RPC."
            )
        return self._handler

    @property
    def single_token_size(self) -> int:
        """Bytes per single token in a KV chunk.

        Used by ``L1Manager``'s maru branch when invoking
        :meth:`get_by_location` to materialize a ``MemoryObj`` for a
        partial chunk.

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        if self._single_token_size == 0:
            raise RuntimeError(
                "MaruMemoryAllocator.single_token_size accessed before init_layout()."
            )
        return self._single_token_size

    def get_by_location(
        self,
        region_id: int,
        page_index: int,
        actual_size: int,
        single_token_size: Optional[int] = None,
    ) -> Optional[MemoryObj]:
        """Resolve a CXL ``(region_id, page_index)`` to a
        ``MemoryObj``.

        Used during the RETRIEVE lookup phase:
        ``MaruHandler.batch_retrieve`` reports the location and this
        method materialises the pool-resident ``MemoryObj`` (no data
        copy).

        Args:
            region_id: Region id from ``MaruServer``.
            page_index: Page index within the region.
            actual_size: Actual KV chunk size in bytes (may be less
                than a full chunk for trailing partial chunks).
            single_token_size: Bytes-per-token override for partial
                chunks. Defaults to :attr:`single_token_size`.

        Returns:
            ``MemoryObj`` resolving the CXL page, or ``None`` if the
            location is no longer valid (e.g. region not mapped).

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        self._require_initialized("get_by_location")
        if single_token_size is None:
            single_token_size = self._single_token_size
        return self._cxl_adapter.get_by_location(  # type: ignore[union-attr]
            region_id=region_id,
            page_index=page_index,
            actual_size=actual_size,
            single_token_size=single_token_size,
        )

    def create_store_handle(self, memory_obj: MemoryObj) -> Any:
        """Reconstruct an ``AllocHandle`` from a ``MemoryObj`` for use
        with ``MaruHandler.batch_store``.

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        self._require_initialized("create_store_handle")
        return self._cxl_adapter.create_store_handle(memory_obj)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # MemoryAllocatorInterface
    # ------------------------------------------------------------------

    def allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate a single CXL-backed ``MemoryObj``.

        ``CxlMemoryAdapter`` uses the canonical shapes/dtypes/fmt
        fixed at :meth:`init_layout` time; the arguments here are
        accepted for interface compatibility but the pool's metadata
        is authoritative.

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        self._require_initialized("allocate")
        return self._cxl_adapter.allocate(shapes, dtypes, fmt, allocator_type)  # type: ignore[union-attr]

    def batched_allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate ``batch_size`` CXL-backed ``MemoryObj`` instances.

        Raises:
            RuntimeError: If :meth:`init_layout` has not yet been
                called.
        """
        self._require_initialized("batched_allocate")
        return self._cxl_adapter.batched_allocate(  # type: ignore[union-attr]
            shapes, dtypes, batch_size, fmt, allocator_type
        )

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """No-op. CXL lifecycle owned by MaruServer."""
        return

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """No-op. CXL lifecycle owned by MaruServer."""
        return

    def close(self) -> None:
        """Close the underlying ``CxlMemoryAdapter`` and
        ``MaruHandler`` if they were ever built.

        Best-effort: errors during close are logged but do not
        propagate. Safe to call before :meth:`init_layout` — both
        underlying objects are ``None`` in that case and the call is
        a no-op.
        """
        if self._cxl_adapter is not None:
            try:
                self._cxl_adapter.close()
            except Exception:
                logger.exception(
                    "[MaruMemoryAllocator] CxlMemoryAdapter.close() failed"
                )
            self._cxl_adapter = None
        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:
                logger.exception("[MaruMemoryAllocator] MaruHandler.close() failed")
            self._handler = None

    def memcheck(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_initialized(self, op: str) -> None:
        if self._cxl_adapter is None:
            raise RuntimeError(
                f"MaruMemoryAllocator.{op} called before init_layout(); "
                f"call init_layout(shapes, dtypes, fmt, chunk_size_in_tokens) "
                f"first (typically from MPCacheEngine.register_kv_cache)."
            )


def _compute_full_chunk_size_bytes(
    shapes: List[torch.Size], dtypes: List[torch.dtype]
) -> int:
    """Total bytes for one full KV chunk across all layer groups.

    Args:
        shapes: Per-layer-group shapes.
        dtypes: Per-layer-group dtypes (must align with ``shapes``).

    Returns:
        ``sum(shape.numel() * dtype.itemsize)``.

    Raises:
        ValueError: If ``shapes`` and ``dtypes`` differ in length.
    """
    if len(shapes) != len(dtypes):
        raise ValueError(
            f"shapes and dtypes must have the same length, "
            f"got {len(shapes)} and {len(dtypes)}"
        )
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )
