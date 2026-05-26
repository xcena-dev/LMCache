# SPDX-License-Identifier: Apache-2.0

"""Maru-mode dispatch logic for L1Manager.

This module isolates the maru-specific behaviour ``L1Manager`` would
otherwise carry inline. ``L1Manager`` constructs a
:class:`MaruL1Dispatcher` when it detects a ``MaruMemoryAllocator``
and forwards each public method to it.

The dispatcher owns:

- ``MaruMemoryAllocator`` reference — for the ``handler`` property and
  the ``get_by_location`` / ``create_store_handle`` extension methods.
- ``L1MemoryManager`` reference — used by :meth:`reserve_write` to
  drive allocation and by :meth:`report_status` to read usage stats.
- ``_pending_read_memobjs`` side channel — populated in
  :meth:`reserve_read` and drained by :meth:`unsafe_read` /
  :meth:`finish_read`.

Thread safety: the dispatcher assumes the caller holds the
``L1Manager`` lock (the public methods on L1Manager are wrapped with
``@l1_mgr_synchronized``). The side-channel dict is therefore not
guarded by a separate lock here.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import MaruMemoryAllocator
    from lmcache.v1.distributed.memory_manager import L1MemoryManager

logger = init_logger(__name__)


def object_key_to_string(key: ObjectKey) -> str:
    """Stable string representation of ``ObjectKey`` for ``MaruHandler``
    RPCs.

    The format mirrors the encoding used by other L2 adapters
    (``model@kv_rank_hex@chunk_hash_hex[@salt]``) so KV index entries
    are inter-operable with adapters that might query the same
    MaruServer instance through the L2 path.

    Args:
        key: The object key to encode.

    Returns:
        ``"<model>@<kv_rank:08x>@<chunk_hash_hex>[@<salt>]"``.
    """
    base = f"{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


class MaruL1Dispatcher:
    """Encapsulates maru-mode dispatch for L1Manager.

    Each method maps 1:1 to the corresponding ``L1Manager`` public
    method. The dispatcher is constructed only when the L1 allocator
    is a :class:`MaruMemoryAllocator` — see
    :class:`L1Manager.__init__`.
    """

    def __init__(
        self,
        allocator: "MaruMemoryAllocator",
        memory_manager: "L1MemoryManager",
        write_ttl_seconds: int,
        read_ttl_seconds: int,
    ) -> None:
        self._allocator = allocator
        self._memory_manager = memory_manager
        self._write_ttl_seconds = write_ttl_seconds
        self._read_ttl_seconds = read_ttl_seconds
        self._pending_read_memobjs: dict[ObjectKey, MemoryObj] = {}

    @property
    def handler(self) -> Any:
        """The connected ``MaruHandler``.

        Resolves through ``MaruMemoryAllocator.handler``, which raises
        if the allocator's ``init_layout`` has not been called. On the
        engine hot path that ordering is guaranteed by
        ``MPCacheEngine.register_kv_cache`` running before any
        ``store`` / ``lookup`` RPC.
        """
        return self._allocator.handler

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def reserve_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Pin + retrieve + stage MemoryObjs in the side channel.

        ``MaruHandler.batch_pin`` has prefix-stop semantics — it only
        pins the contiguous prefix of existing keys. We then resolve
        each pinned key via ``batch_retrieve`` + ``get_by_location``
        to materialise a ``MemoryObj`` pointing at the existing CXL
        page (no data copy). The resolved ``MemoryObj`` is staged in
        ``self._pending_read_memobjs`` so the subsequent
        ``unsafe_read`` can return it.

        If a pinned key cannot be resolved (race between pin and
        retrieve), we unpin the unused tail to keep MaruServer's
        ``pin_count`` accurate.
        """
        handler = self.handler
        key_strs = [object_key_to_string(k) for k in keys]
        try:
            pin_results = handler.batch_pin(key_strs)
        except Exception:
            logger.exception("MaruHandler.batch_pin failed for %d keys", len(keys))
            return {k: (L1Error.KEY_NOT_EXIST, None) for k in keys}

        num_pinned = 0
        for ok in pin_results:
            if not ok:
                break
            num_pinned += 1

        ret: dict[ObjectKey, L1OperationResult] = {
            k: (L1Error.KEY_NOT_EXIST, None) for k in keys
        }
        if num_pinned == 0:
            return ret

        try:
            mem_infos = handler.batch_retrieve(key_strs[:num_pinned])
        except Exception:
            logger.exception(
                "MaruHandler.batch_retrieve failed for %d keys", num_pinned
            )
            # Roll back the pins so MaruServer's refcount stays consistent.
            try:
                handler.batch_unpin(key_strs[:num_pinned])
            except Exception:
                logger.exception(
                    "MaruHandler.batch_unpin rollback failed for %d keys",
                    num_pinned,
                )
            return ret

        resolved = 0
        for k, mi in zip(keys[:num_pinned], mem_infos, strict=False):
            if mi is None:
                # Race between pin and retrieve — treat this and all
                # subsequent keys as miss to preserve prefix semantics.
                break
            mem_obj = self._allocator.get_by_location(
                region_id=mi.region_id,
                page_index=mi.page_index,
                actual_size=len(mi.view),
            )
            if mem_obj is None:
                break
            self._pending_read_memobjs[k] = mem_obj
            ret[k] = (L1Error.SUCCESS, mem_obj)
            resolved += 1

        if resolved < num_pinned:
            extras = key_strs[resolved:num_pinned]
            try:
                handler.batch_unpin(extras)
            except Exception:
                logger.exception(
                    "MaruHandler.batch_unpin (reconciliation) failed for %d keys",
                    len(extras),
                )
        return ret

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Look up MemoryObjs staged by :meth:`reserve_read`."""
        ret: dict[ObjectKey, L1OperationResult] = {}
        for k in keys:
            mem_obj = self._pending_read_memobjs.get(k)
            if mem_obj is None:
                ret[k] = (L1Error.KEY_NOT_EXIST, None)
            else:
                ret[k] = (L1Error.SUCCESS, mem_obj)
        return ret

    def finish_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Drop side-channel entries and ``batch_unpin``."""
        handler = self.handler

        ret: dict[ObjectKey, L1Error] = {}
        to_unpin: list[str] = []
        for k in keys:
            if self._pending_read_memobjs.pop(k, None) is not None:
                to_unpin.append(object_key_to_string(k))
                ret[k] = L1Error.SUCCESS
            else:
                ret[k] = L1Error.KEY_NOT_EXIST

        if to_unpin:
            try:
                handler.batch_unpin(to_unpin)
            except Exception:
                logger.exception(
                    "MaruHandler.batch_unpin failed in finish_read for %d keys",
                    len(to_unpin),
                )
        return ret

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Allocate CXL ``MemoryObj``s for ``keys``.

        No in-process dict / TTLLock / state machine is used. The
        engine takes the returned ``MemoryObj``s, runs cudaMemcpy
        into their CXL-backed ``data_ptr``, then hands them back via
        :meth:`finish_write` (which issues ``batch_store``).

        ``is_temporary`` and ``mode`` are accepted for interface
        compatibility but have no effect in maru mode — the maru flow
        only uses ``mode="new"``.
        """
        del is_temporary, mode  # unused in maru mode

        ret: dict[ObjectKey, L1OperationResult] = {}
        if not keys:
            return ret

        err, allocated_objs = self._memory_manager.allocate(layout_desc, len(keys))
        if err != L1Error.SUCCESS:
            for k in keys:
                ret[k] = (L1Error.OUT_OF_MEMORY, None)
            return ret

        for k, obj in zip(keys, allocated_objs, strict=False):
            ret[k] = (L1Error.SUCCESS, obj)
        return ret

    def finish_write(
        self,
        keys: list[ObjectKey],
        memory_objs: Optional[list[MemoryObj]],
    ) -> dict[ObjectKey, L1Error]:
        """Register KVs with MaruServer via ``batch_store``.

        ``batch_store`` performs dup-skip + auto-free transparently:
        keys that already exist have their newly-allocated CXL page
        returned to the pool. Both "newly registered" and
        "skipped because already present" are functional successes.
        """
        handler = self.handler

        if memory_objs is None or len(memory_objs) != len(keys):
            actual = 0 if memory_objs is None else len(memory_objs)
            logger.error(
                "Maru finish_write requires memory_objs matching keys "
                "(keys=%d, memory_objs=%d)",
                len(keys),
                actual,
            )
            return {k: L1Error.KEY_IN_WRONG_STATE for k in keys}

        key_strs = [object_key_to_string(k) for k in keys]
        try:
            handles = [self._allocator.create_store_handle(mo) for mo in memory_objs]
        except Exception:
            logger.exception(
                "create_store_handle failed for %d MemoryObjs", len(memory_objs)
            )
            return {k: L1Error.KEY_IN_WRONG_STATE for k in keys}

        try:
            results = handler.batch_store(key_strs, handles)
        except Exception:
            logger.exception("MaruHandler.batch_store failed for %d keys", len(keys))
            return {k: L1Error.KEY_IN_WRONG_STATE for k in keys}

        ret: dict[ObjectKey, L1Error] = {}
        for k, ok in zip(keys, results, strict=False):
            ret[k] = L1Error.SUCCESS if ok else L1Error.KEY_IN_WRONG_STATE
        return ret

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, L1OperationResult]:
        """Defensive no-op for the atomic write-to-read transition.

        The maru flow stages MemoryObjs in the side channel during
        :meth:`reserve_read` rather than going through the
        ``reserve_write(is_temporary=True)`` → ``submit_load_task`` →
        ``finish_write_and_reserve_read`` sequence used by other
        backends. Return ``SUCCESS`` for already-staged keys and
        ``KEY_NOT_EXIST`` otherwise so any defensive caller still
        sees a useful answer.
        """
        ret: dict[ObjectKey, L1OperationResult] = {}
        for k in keys:
            mem_obj = self._pending_read_memobjs.get(k)
            if mem_obj is None:
                ret[k] = (L1Error.KEY_NOT_EXIST, None)
            else:
                ret[k] = (L1Error.SUCCESS, mem_obj)
        return ret

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Forward to ``MaruHandler.delete`` per key.

        ``MaruHandler.delete`` returns ``False`` when the key is
        pinned or missing — the API conflates the two, so we report
        the softer ``KEY_NOT_EXIST`` (callers retrying after
        ``KEY_IS_LOCKED`` would quickly hit it again anyway).
        """
        handler = self.handler

        ret: dict[ObjectKey, L1Error] = {}
        for k in keys:
            key_str = object_key_to_string(k)
            try:
                ok = handler.delete(key_str)
            except Exception:
                logger.exception("MaruHandler.delete failed for key=%s", key_str)
                ret[k] = L1Error.KEY_IN_WRONG_STATE
                continue
            ret[k] = L1Error.SUCCESS if ok else L1Error.KEY_NOT_EXIST
        return ret

    def clear(self, force: bool) -> None:
        """Drop staged side-channel entries only.

        The CXL pool itself is owned by ``MaruServer`` and is never
        wiped by the L1 layer — ``force=True`` only affects the
        in-process read-side bookkeeping. Server-side wipes go
        through explicit ``MaruHandler.delete`` calls or MaruServer's
        own lifecycle.
        """
        if force:
            logger.warning(
                "L1Manager (maru): force-clear drops %d pending read "
                "MemoryObjs but does NOT touch MaruServer.",
                len(self._pending_read_memobjs),
            )
        self._pending_read_memobjs.clear()

    def report_status(self) -> dict:
        """Maru-flavoured status snapshot."""
        used, total = self._memory_manager.get_memory_usage()
        return {
            "is_healthy": True,
            "backend": "maru",
            "total_object_count": 0,
            "write_locked_count": 0,
            "read_locked_count": 0,
            "temporary_count": 0,
            "pending_read_memobjs": len(self._pending_read_memobjs),
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }
