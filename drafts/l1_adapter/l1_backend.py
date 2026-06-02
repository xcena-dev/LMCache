# SPDX-License-Identifier: Apache-2.0
"""Integration point 2 (alloc/lookup/usage) draft — the L1Backend protocol.

REVIEW DRAFT. Not imported anywhere.

`L1Manager` currently takes a concrete ``gds_backend: GdsL1Backend | None`` and
imports ``GdsL1Backend`` directly (PR #3420). This draft extracts exactly the
surface `L1Manager` uses into a Protocol, so the manager depends on an
abstraction and any GPU-DMA-able L1 medium (GDS, CXL-pool, DAX, Maru) can
implement it.

The DMA itself is NOT here — that lives on the backend's allocator
(``dma_to_gpu`` / ``dma_from_gpu``, see gpu_ops_after.py). This protocol is the
*management* face: where bytes are allocated, what is durably resident, and how
much is used. The two together are what "an L1 backend" means to the core.

Proposed home: ``lmcache/v1/distributed/l1_backend.py``.
"""

# Standard
from typing import Optional, Protocol, runtime_checkable

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import MemoryObj


@runtime_checkable
class L1Backend(Protocol):
    """A GPU-DMA-able L1 medium that owns its own allocation and durable index.

    A medium is L1 when an xPU can DMA to/from it without staging (issue #3262):
    GDS (NVMe via cuFile P2P DMA), CXL-pooled memory, DAX, etc. Such a backend
    differs from the default pinned-DRAM path in three management aspects, which
    make up this protocol:

    1. **Write side** — it mints its own backend-anchored ``MemoryObj`` instead
       of allocating from the CPU pinned slab (:meth:`create_memory_obj`).
    2. **Read side** — it can resurrect an entry that is durably resident on the
       medium but absent from the in-memory index, i.e. fill-on-miss
       (:meth:`create_memory_obj_from_index`).
    3. **Accounting / lifecycle** — it reports its own usage to the eviction
       controller (:meth:`get_memory_usage`) and tears down on :meth:`close`.

    ``L1Manager`` holds at most one backend (see open question #1 in the design
    doc about supporting more than one at once). With no backend attached, the
    manager runs the pinned-DRAM path byte-for-byte unchanged.

    The objects returned here carry the right metadata and have their
    ``parent()`` set to the backend's DMA-able allocator, so the eventual GPU
    transfer routes through that allocator's ``dma_to_gpu`` / ``dma_from_gpu``.
    No bytes move inside these methods.
    """

    def create_memory_obj(
        self,
        key: ObjectKey,
        layout: MemoryLayoutDesc,
    ) -> MemoryObj:
        """Mint a backend-anchored MemoryObj for a ``reserve_write``.

        Called by ``L1Manager.reserve_write`` per key being written when a
        backend is attached. The returned object's ``parent()`` must be the
        backend's allocator so the eventual D2H DMA goes through the medium
        (e.g. cuFile write). The on-medium region is materialized when the
        caller performs the D2H DMA.

        Args:
            key: The cache key being reserved for write.
            layout: Tensor layout (shape/dtype/format) of the chunk.

        Returns:
            A backend-anchored MemoryObj owned by this backend's allocator.
        """
        ...

    def create_memory_obj_from_index(self, key: ObjectKey) -> Optional[MemoryObj]:
        """Fill-on-miss: return a MemoryObj if ``key`` is durably resident.

        Called by ``L1Manager.reserve_read`` when the in-memory index misses.
        If the medium has ``key`` resident (e.g. in its persistent index),
        return a MemoryObj that makes it reachable; the data is DMA'd in later
        via ``dma_to_gpu``. Return ``None`` if not resident, in which case the
        manager reports ``KEY_NOT_EXIST``.

        Args:
            key: The cache key looked up on a read miss.

        Returns:
            A backend-anchored MemoryObj, or ``None`` if not resident.
        """
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` for this medium.

        Replaces the pinned-slab usage signal when a backend is attached. The
        eviction controller consumes ``(used, total)`` agnostically, so the
        source swap is transparent to it.

        Returns:
            A tuple ``(used_bytes, total_bytes)``.
        """
        ...

    def close(self) -> None:
        """Release backend resources (file handles, registrations, loops).

        Called from ``L1Manager.close`` after the in-memory manager is closed.
        """
        ...
