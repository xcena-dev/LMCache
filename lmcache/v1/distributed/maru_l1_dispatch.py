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
  :meth:`finish_read`. Each entry is a :class:`_PendingRead` carrying
  the staged ``MemoryObj`` plus a ``refcount``, so overlapping reads of
  the SAME key share one staged object (same CXL page) while each still
  balances its own remote pin with exactly one remote unpin.

Thread safety: the dispatcher holds no lock of its own. Its owner,
:class:`~lmcache.v1.distributed.maru_l1_manager.MaruL1Manager`, provides
a ``threading.Lock`` and serialises every public method that touches
these side channels (the maru sibling of ``@l1_mgr_synchronized``), so
all dispatcher state is only ever accessed under that lock.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


@dataclass
class _PendingRead:
    """A staged read entry with an outstanding-pin reference count.

    Overlapping ``reserve_read`` calls for the SAME key each perform their
    own remote pin (correct: N reserves == N remote pins) but share a
    single staged ``MemoryObj`` (they all resolve to the same CXL page).
    ``refcount`` tracks how many ``finish_read`` calls are still
    outstanding; :meth:`MaruL1Dispatcher.finish_read` issues one remote
    unpin per call and drops the entry only when the count reaches zero,
    so N finishes == N unpins and the remote ``pin_count`` always
    balances.

    Attributes:
        mem_obj: The staged ``MemoryObj`` resolving the CXL page.
        refcount: Number of outstanding remote pins not yet unpinned.
    """

    mem_obj: MemoryObj
    refcount: int


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
        self._pending_read_memobjs: dict[ObjectKey, _PendingRead] = {}
        # Write-side channel: reserve_write stashes reserved MemoryObjs here so
        # the keys-only finish_write drain can recover them (mirrors
        # _pending_read_memobjs). finish_write pops them. See reserve_write for
        # the store-failure leak caveat; the count is surfaced in report_status.
        self._pending_write_memobjs: dict[ObjectKey, MemoryObj] = {}

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

        Every successful ``reserve_read`` of a key performs its own
        remote pin (N reserves == N remote pins, matched by N unpins in
        :meth:`finish_read`). When a key is already staged by an
        overlapping read, the staged :class:`_PendingRead` has its
        ``refcount`` incremented and the existing ``MemoryObj`` is kept
        (same CXL page); the freshly materialised view is discarded
        (``get_by_location`` allocates no pool slot and ``free`` is a
        no-op, so there is no local leak).

        If a pinned key cannot be resolved (race between pin and
        retrieve), we unpin the unused tail to keep MaruServer's
        ``pin_count`` accurate. The manager strips ``extra_count`` before
        delegating here — maru does no TP per-key read-lock counting (a
        documented delta from :class:`L1Manager`).
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
            staged = self._pending_read_memobjs.get(k)
            if staged is not None:
                # Already staged by an overlapping reserve_read: this call
                # issued its own remote pin, so bump the refcount and keep
                # the existing staged MemoryObj (same CXL page). Discard the
                # freshly materialised view (no pool slot, free() no-op).
                staged.refcount += 1
                ret[k] = (L1Error.SUCCESS, staged.mem_obj)
            else:
                self._pending_read_memobjs[k] = _PendingRead(
                    mem_obj=mem_obj, refcount=1
                )
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
        """Look up MemoryObjs staged by :meth:`reserve_read`.

        Returns the staged ``MemoryObj`` regardless of its refcount, so a
        read issued between two overlapping :meth:`finish_read` calls still
        resolves as long as at least one pin remains outstanding.
        """
        ret: dict[ObjectKey, L1OperationResult] = {}
        for k in keys:
            entry = self._pending_read_memobjs.get(k)
            if entry is None:
                ret[k] = (L1Error.KEY_NOT_EXIST, None)
            else:
                ret[k] = (L1Error.SUCCESS, entry.mem_obj)
        return ret

    def finish_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Decrement staged refcounts and ``batch_unpin`` once per key.

        Each ``finish_read`` of a staged key issues exactly one remote
        unpin, balancing the one remote pin its matching ``reserve_read``
        performed (N reserves -> N pins -> N finishes -> N unpins). The
        staged :class:`_PendingRead` is dropped only when its refcount
        reaches zero, so an overlapping reader still sees the object until
        the last finish. Keys that are not staged report ``KEY_NOT_EXIST``
        and are not unpinned (preserving the original miss behaviour).
        """
        handler = self.handler

        ret: dict[ObjectKey, L1Error] = {}
        to_unpin: list[str] = []
        for k in keys:
            entry = self._pending_read_memobjs.get(k)
            if entry is not None:
                entry.refcount -= 1
                to_unpin.append(object_key_to_string(k))
                if entry.refcount <= 0:
                    del self._pending_read_memobjs[k]
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
            entry = self._pending_read_memobjs.get(k)
            if entry is None:
                ret[k] = (L1Error.KEY_NOT_EXIST, None)
            else:
                ret[k] = (L1Error.SUCCESS, entry.mem_obj)
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
        """Drop staged side-channel entries, balancing remote pins.

        Each staged read holds ``refcount`` outstanding remote pins, so
        before dropping the read side channel we ``batch_unpin`` once per
        remaining refcount (not once per key) to keep MaruServer's
        ``pin_count`` balanced when reads are force-dropped rather than
        completed through :meth:`finish_read`. This releases pins only —
        the CXL pool and the stored KV data are owned by ``MaruServer``
        and are never wiped by the L1 layer (data deletion goes through
        explicit ``MaruHandler.delete``). ``force`` only controls logging;
        both side channels are always cleared.
        """
        to_unpin: list[str] = []
        for k, entry in self._pending_read_memobjs.items():
            to_unpin.extend([object_key_to_string(k)] * entry.refcount)

        if force:
            logger.warning(
                "L1Manager (maru): force-clear drops %d pending read "
                "(%d outstanding pins) + %d pending write MemoryObjs; the "
                "pins are released on MaruServer but stored data is NOT "
                "deleted.",
                len(self._pending_read_memobjs),
                len(to_unpin),
                len(self._pending_write_memobjs),
            )

        if to_unpin:
            try:
                self.handler.batch_unpin(to_unpin)
            except Exception:
                logger.exception(
                    "MaruHandler.batch_unpin failed in clear for %d pins",
                    len(to_unpin),
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
