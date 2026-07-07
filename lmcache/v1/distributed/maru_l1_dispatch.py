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
import os
import threading

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
        # Write-side channel: reserve_write stashes reserved MemoryObjs here so
        # the keys-only finish_write drain can recover them (mirrors
        # _pending_read_memobjs). finish_write pops them. See reserve_write for
        # the store-failure leak caveat; the count is surfaced in report_status.
        self._pending_write_memobjs: dict[ObjectKey, MemoryObj] = {}

        # Lookahead prefetch (record-order -> prefetch the batch k ahead).
        # 0 = disabled (current reactive behavior). Relies on a replayed
        # retrieve-batch order (e.g. the cache_hit pass of repeat_count>=2):
        # batches are matched by their exact key sequence. MP maru-L1 path
        # mirror of MaruBackend._lookahead_prefetch (single-process).
        self._lookahead_depth: int = int(
            os.environ.get("MARU_GAIA_LOOKAHEAD_DEPTH", "0")
        )
        self._lookahead_lock = threading.Lock()
        self._seen_order: list[list[str]] = []
        self._batch_index: dict[tuple[str, ...], int] = {}
        if self._lookahead_depth > 0:
            logger.info(
                "[Maru] L1 lookahead prefetch enabled (MARU_GAIA_LOOKAHEAD_DEPTH=%d)",
                self._lookahead_depth,
            )

        # Lookup-time prefetch (issue a Gaia prefetch of the maru/L1 hit prefix
        # at the admission-stage lookup -- inside :meth:`reserve_read`, right
        # after ``batch_pin`` establishes the prefix and before the reserved KV
        # is copied out of its CXL page). The request's admission wait then
        # doubles as the SSD->CXL fill window, so the later copy-out reads warm
        # CXL DRAM instead of racing the fill. Independent of lookahead depth;
        # 0/unset = off (reactive). MP maru-L1 mirror of
        # MaruBackend._lookup_prefetch (single-process).
        self._prefetch_on_lookup: bool = (
            os.environ.get("MARU_GAIA_PREFETCH_ON_LOOKUP", "0") == "1"
        )
        if self._prefetch_on_lookup:
            logger.info(
                "[Maru] L1 lookup-time prefetch enabled "
                "(MARU_GAIA_PREFETCH_ON_LOOKUP=1)"
            )

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

    def _lookahead_prefetch(self, key_strs: list[str]) -> None:
        """Record the retrieve-batch order and, on a replayed order, prefetch
        the batch ``MARU_GAIA_LOOKAHEAD_DEPTH`` positions ahead.

        First sighting of a batch (by its exact key sequence) is only
        recorded. When the same batch is seen again -- e.g. the cache_hit
        pass of a ``repeat_count>=2`` benchmark -- the batch ``depth``
        positions later in the recorded order is prefetched via
        ``MaruHandler.prefetch_batch`` so it is warm in CXL DRAM by the time
        it is retrieved. No-op when depth is 0 or the batch is empty.

        Args:
            key_strs: The string keys of the batch currently being retrieved.
        """
        depth = self._lookahead_depth
        if depth <= 0 or not key_strs:
            return
        sig = tuple(key_strs)
        target_keys: Optional[list[str]] = None
        with self._lookahead_lock:
            idx = self._batch_index.get(sig)
            if idx is None:
                # First sighting -- recording pass; nothing to prefetch yet.
                self._batch_index[sig] = len(self._seen_order)
                self._seen_order.append(key_strs)
                return
            target = idx + depth
            if 0 <= target < len(self._seen_order):
                target_keys = self._seen_order[target]
        if target_keys is not None:
            try:
                self.handler.prefetch_batch(target_keys)
            except Exception:
                logger.warning(
                    "[Maru] L1 lookahead prefetch_batch failed", exc_info=True
                )

    def _lookup_prefetch(self, key_strs: list[str], num_hit: int) -> None:
        """Issue a Gaia prefetch for the maru/L1 hit prefix at lookup time.

        Called from :meth:`reserve_read` (the admission-stage lookup that
        ``StorageManager.submit_prefetch_task`` drives) once ``batch_pin`` has
        established how many contiguous prefix keys exist in the maru/CXL
        tier. Firing the prefetch here -- before the reserved KV is copied out
        of its CXL page -- lets the request's admission wait double as the
        SSD->CXL fill window, so the later read hits warm CXL DRAM instead of
        racing the fill. ``key_strs`` are the exact ``object_key_to_string``
        encodings already used for ``batch_pin`` / ``batch_retrieve``, so the
        prefetched regions match the eventual read. No-op when lookup-time
        prefetch is disabled or nothing hit.

        Args:
            key_strs: The looked-up keys (already string-encoded) in prefix
                order.
            num_hit: Number of contiguous prefix keys that exist (from the
                ``batch_pin`` check); only this prefix is prefetched.
        """
        if not self._prefetch_on_lookup or num_hit <= 0:
            return
        try:
            self.handler.prefetch_batch(key_strs[:num_hit])
        except Exception:
            logger.warning("[Maru] L1 lookup-time prefetch_batch failed", exc_info=True)

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
        self._lookahead_prefetch(key_strs)
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

        # Lookup-time prefetch: warm the CXL DRAM for the hit prefix now (during
        # the request's admission wait) so the later copy-out reads warm memory.
        # No-op unless MARU_GAIA_PREFETCH_ON_LOOKUP=1 and the prefix hit.
        self._lookup_prefetch(key_strs, num_pinned)

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
            # Stash for the keys-only finish_write drain to recover (the post-
            # merge completion path carries only keys). finish_write pops these.
            #
            # KNOWN LIMITATION (deferred): if the engine's store FAILS after
            # reserve_write, finish_write is never submitted (see
            # lmcache_driven_transfer.store's fail-closed ``finally``), so this
            # entry is never popped AND the underlying CXL page is never freed.
            # maru has NO orphan reclamation: ``handler.alloc`` consumes an
            # ``OwnedRegionManager`` slot that is released only by ``batch_store``
            # (dup / register-fail) or ``delete``/eviction -- none of which run
            # on a failed store -- and ``MaruMemoryAllocator.free`` is a no-op.
            # Accepted for now: store failures are rare and maru L1 eviction is
            # disabled anyway. Proper reclamation (a write-TTL sweep keyed off
            # ``self._write_ttl_seconds``, or an explicit abort path) is left to
            # a dedicated memory-lifecycle design. ``report_status`` surfaces the
            # pending count so any growth is observable.
            self._pending_write_memobjs[k] = obj
        return ret

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Register reserved KVs with MaruServer via ``batch_store``.

        The post-merge completion drain is keys-only, so the
        ``MemoryObj``s reserved in :meth:`reserve_write` are recovered
        from ``_pending_write_memobjs`` (the write-side equivalent of
        ``_pending_read_memobjs``). Every requested key is popped from
        the side channel before returning -- on success, failure, or
        exception -- so the channel never grows on the normal path.

        ``batch_store`` performs dup-skip + auto-free transparently:
        keys that already exist have their newly-allocated CXL page
        returned to the pool. Both "newly registered" and "skipped
        because already present" are functional successes.
        """
        handler = self.handler
        try:
            memory_objs = [self._pending_write_memobjs.get(k) for k in keys]
            present_keys = [
                k for k, mo in zip(keys, memory_objs, strict=False) if mo is not None
            ]
            present_objs = [mo for mo in memory_objs if mo is not None]

            # Keys absent from the side channel were never reserved (or already
            # finished). This should not happen on the normal store path.
            ret: dict[ObjectKey, L1Error] = {
                k: L1Error.KEY_IN_WRONG_STATE
                for k, mo in zip(keys, memory_objs, strict=False)
                if mo is None
            }
            if ret:
                logger.error(
                    "Maru finish_write: %d/%d keys missing from the write side "
                    "channel (never reserved or already finished)",
                    len(ret),
                    len(keys),
                )
            if not present_keys:
                return ret

            key_strs = [object_key_to_string(k) for k in present_keys]
            try:
                handles = [
                    self._allocator.create_store_handle(mo) for mo in present_objs
                ]
            except Exception:
                logger.exception(
                    "create_store_handle failed for %d MemoryObjs", len(present_objs)
                )
                for k in present_keys:
                    ret[k] = L1Error.KEY_IN_WRONG_STATE
                return ret

            try:
                results = handler.batch_store(key_strs, handles)
            except Exception:
                logger.exception(
                    "MaruHandler.batch_store failed for %d keys", len(present_keys)
                )
                for k in present_keys:
                    ret[k] = L1Error.KEY_IN_WRONG_STATE
                return ret

            for k, ok in zip(present_keys, results, strict=False):
                ret[k] = L1Error.SUCCESS if ok else L1Error.KEY_IN_WRONG_STATE
            return ret
        finally:
            for k in keys:
                self._pending_write_memobjs.pop(k, None)

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
                "L1Manager (maru): force-clear drops %d pending read + %d "
                "pending write MemoryObjs but does NOT touch MaruServer.",
                len(self._pending_read_memobjs),
                len(self._pending_write_memobjs),
            )
        self._pending_read_memobjs.clear()
        self._pending_write_memobjs.clear()

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
            "pending_write_memobjs": len(self._pending_write_memobjs),
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }
