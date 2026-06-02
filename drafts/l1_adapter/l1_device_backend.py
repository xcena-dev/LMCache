# SPDX-License-Identifier: Apache-2.0
"""Seam B draft — the L1DeviceBackend protocol.

This is a REVIEW DRAFT. It is not imported anywhere.

`L1Manager` currently takes a concrete ``gds_backend: GdsL1Backend | None`` and
imports ``GdsL1Backend`` directly (PR #3420). This draft extracts exactly the
surface `L1Manager` uses into a Protocol, so the manager depends on an
abstraction and any non-DRAM L1 medium (GDS, CXL-pool, DAX, Maru) can implement
it.

Proposed home: ``lmcache/v1/distributed/l1_device_backend.py``.
"""

# Standard
from typing import Optional, Protocol, runtime_checkable

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import MemoryObj


@runtime_checkable
class L1DeviceBackend(Protocol):
    """A non-DRAM L1 medium that owns its own allocation and durable index.

    An L1 medium qualifies as L1 when an xPU can access it directly without
    staging (issue #3262): GDS (NVMe via cuFile DMA), CXL-pooled memory, DAX,
    etc. Such a backend differs from the default pinned-DRAM path in three ways,
    which together make up this protocol:

    1. **Write side** — it mints its own device-anchored ``MemoryObj`` instead
       of allocating from the CPU pinned slab (:meth:`create_memory_obj`).
    2. **Read side** — it can resurrect an entry that is durably resident on the
       device but not currently in the in-memory index, i.e. fill-on-miss
       (:meth:`create_memory_obj_from_index`).
    3. **Accounting / lifecycle** — it reports its own usage to the eviction
       controller (:meth:`get_memory_usage`) and tears down on
       :meth:`close`.

    ``L1Manager`` holds at most one backend (see open question #1 in the design
    doc about supporting more than one simultaneously). When no backend is
    attached, the manager runs the pinned-DRAM path byte-for-byte unchanged.

    The actual H2D/D2H data movement is NOT part of this protocol — it happens
    through Seam A (the allocator's ``copy_to_gpu`` / ``copy_from_gpu``) when the
    caller does the GPU copy. The objects returned here only need to be
    reachable and carry the right metadata.
    """

    def create_memory_obj(
        self,
        key: ObjectKey,
        layout: MemoryLayoutDesc,
    ) -> MemoryObj:
        """Mint a device-anchored MemoryObj for a ``reserve_write``.

        Called by ``L1Manager.reserve_write`` for each key being written when a
        device backend is attached. The returned object's ``parent()`` must be
        the device's allocator so Seam A routes the eventual D2H copy through the
        device transfer (e.g. cuFile write). No bytes are moved yet; the device
        file/region is materialized when the caller performs the D2H copy.

        Args:
            key: The cache key being reserved for write.
            layout: Tensor layout (shape/dtype/format) of the chunk.

        Returns:
            A device-anchored MemoryObj owned by this backend's allocator.

        Raises:
            L1Error-equivalent: Implementations should signal allocation
                failure in the way L1Manager expects (e.g. raise or return a
                sentinel — to be fixed during integration; PR #3420 returns the
                object and lets allocation fail lazily).
        """
        ...

    def create_memory_obj_from_index(self, key: ObjectKey) -> Optional[MemoryObj]:
        """Fill-on-miss: return a MemoryObj if ``key`` is durably resident.

        Called by ``L1Manager.reserve_read`` when the in-memory index misses.
        If the device has ``key`` resident (e.g. in its persistent slab index),
        return a MemoryObj that makes it reachable; the data is read later via
        the Seam-A copy path. Return ``None`` if the device does not have it,
        in which case the manager reports ``KEY_NOT_EXIST``.

        Args:
            key: The cache key looked up on a read miss.

        Returns:
            A device-anchored MemoryObj, or ``None`` if not resident.
        """
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` for this device.

        Replaces the pinned-slab usage signal when a backend is attached. The
        eviction controller consumes ``(used, total)`` agnostically, so the
        source swap is transparent to it.

        Returns:
            A tuple ``(used_bytes, total_bytes)``.
        """
        ...

    def close(self) -> None:
        """Release device resources (file handles, registrations, loops).

        Called from ``L1Manager.close`` after the in-memory manager is closed.
        """
        ...
