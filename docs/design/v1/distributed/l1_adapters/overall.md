# L1 Adapter Design (proposal / draft)

> Status: **DRAFT for discussion** — see issue #3262 (Distributed MP mode RFC)
> and PR #3420 (GDS L1 Draft). This document proposes a generic L1 device-backend
> abstraction so that GDS, CXL-pooled memory, DAX, and similar "GPU-directly-
> accessible" media plug into MP-mode L1 the same way, instead of each one adding
> a device-specific branch into `L1Manager` and `gpu_ops`.

## Motivation

In issue #3262 the working definition of L1 settled on:

> **L1 = memory that an xPU can access directly, without staging.**

By that definition pinned CPU DRAM is only one instance of L1. GDS (NVMe via
cuFile DMA), CXL-pooled memory, and DAX all qualify. PR #3420 is landing the
first non-DRAM L1 (GDS), and XCENA wants to bring a CXL-pooled Maru backend into
MP mode as a second one.

The problem is **how** a new L1 medium is added today. PR #3420 wires GDS in as
a device-specific bolt-on:

- `lmcache/v1/gpu_connector/gpu_ops.py` gains an
  `isinstance(parent, GdsScratchAllocator)` branch next to the existing
  `isinstance(parent, LazyMemoryAllocator)` branch, and imports `gds_l1`
  directly.
- `lmcache/v1/distributed/l1_manager.py` takes a concrete
  `gds_backend: GdsL1Backend | None` constructor argument, imports
  `GdsL1Backend` directly, and sprinkles `if self._gds_backend is not None:`
  branches through `reserve_read`, `reserve_write`, `get_memory_usage`,
  `get_l1_memory_desc`, and `close`.

This works for one device, but it does not scale: **every new L1 medium (CXL,
DAX, Maru) would add another `isinstance` branch in `gpu_ops` and another
`xxx_backend` hook + scattered `if` branches in `L1Manager`.** The core L1 path
becomes a switchboard of device-specific conditionals, and "adding a new L1
device is easy" (issue #3262) stops being true.

This is the same problem the L2 layer already solved with **L2 adapters**
(`docs/design/v1/distributed/l2_adapters/`). This proposal does the analogous
thing for L1.

## Two seams to generalize

There are exactly two places where device-specific knowledge leaks into the
core today. We generalize each independently; either can land on its own.

### Seam A — `gpu_ops` H2D/D2H dispatch → allocator polymorphism

`lmcache_memcpy_async_h2d` / `_d2h` currently `isinstance`-dispatch on the
`MemoryObj`'s parent allocator. Instead, push the copy into a polymorphic method
on `MemoryAllocatorInterface`:

```python
class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    ...
    def copy_to_gpu(self, memory_obj: MemoryObj, gpu_buffer: torch.Tensor) -> None:
        """Copy this allocator's MemoryObj into a GPU buffer (H2D).

        Default: a stream-ordered cudaMemcpyAsync from the object's
        host tensor. Allocators backed by a non-DRAM medium (lazy-pinned,
        GDS/cuFile, CXL, DAX) override this with their own transfer.
        """
        # default = current `gpu_buffer.copy_(src_tensor...)` path

    def copy_from_gpu(self, gpu_buffer: torch.Tensor, memory_obj: MemoryObj) -> None:
        """Copy a GPU buffer into this allocator's MemoryObj (D2H)."""
        # default = current `dst_tensor.copy_(gpu_buffer...)` path
```

`gpu_ops` then becomes:

```python
def lmcache_memcpy_async_h2d(memory_obj, gpu_buffer):
    _check_sizes(memory_obj, gpu_buffer)
    memory_obj.parent().copy_to_gpu(memory_obj, gpu_buffer)
```

- `LazyMemoryAllocator` overrides with its `lmc_ops.lmcache_memcpy_async` path
  (identical bytes to today).
- `GdsScratchAllocator` overrides with `cufile_read_into` / `cufile_write_from`
  (PR #3420's existing methods — just moved behind the interface).
- A future `CxlAllocator` / `DaxAllocator` overrides with its own transfer.

Net effect: **no device imports in `gpu_ops`, no `isinstance` chain.** Existing
behavior is byte-for-byte preserved; this is a pure refactor and can merge
independently of any device.

### Seam B — `L1Manager` device hook → `L1DeviceBackend` protocol

Extract the GDS-specific hook into a small protocol that `L1Manager` depends on
abstractly. The surface is exactly what PR #3420 already needs from
`GdsL1Backend`:

```python
class L1DeviceBackend(Protocol):
    """A non-DRAM L1 medium that owns its own allocation + durable index.

    L1Manager holds at most one of these. When present, it replaces the
    pinned-slab allocation path on the write side, supplies fill-on-miss on
    the read side, and reports usage/lifecycle.
    """

    def create_memory_obj(
        self, key: ObjectKey, layout: MemoryLayoutDesc
    ) -> MemoryObj:
        """Mint a device-anchored MemoryObj for a reserve_write."""

    def create_memory_obj_from_index(self, key: ObjectKey) -> MemoryObj | None:
        """Fill-on-miss: return a MemoryObj if `key` is durably resident on
        this device, else None. The actual data movement happens later through
        the Seam-A copy path; this only makes the object reachable."""

    def get_memory_usage(self) -> tuple[int, int]:
        """(used_bytes, total_bytes) for this device — feeds eviction."""

    def close(self) -> None: ...
```

`L1Manager.__init__` then takes `device_backend: L1DeviceBackend | None`
(typed by the protocol, no concrete import), and the existing
`if self._gds_backend is not None:` branches become
`if self._device_backend is not None:` — unchanged logic, generalized type.

GDS becomes the **first** `L1DeviceBackend` implementation; CXL-pooled Maru
becomes the **second**; DAX (#3057) a natural third.

## What this does *not* change

- The `MemoryObj` subclassing model (`GdsMemoryObj(MemoryObj)`) and the existing
  ~13 `MemoryAllocatorInterface` implementations are untouched — GDS already
  reuses them correctly.
- The CPU-pinned DRAM path stays the default and byte-for-byte unchanged.
- MP server / `StorageManager` data path is unaffected (it already routes
  through `gpu_ops`).

## Open questions (for #3262 / PR #3420)

1. Can `L1Manager` ever hold **more than one** device backend (e.g. local DRAM +
   shared CXL pool at once), or is one-at-a-time sufficient for now? (The user's
   "Local L1 vs Shared L1" split in #3262 suggests eventually >1.)
2. NIXL co-tenancy with a device-registered VRAM region — `get_l1_memory_desc`
   currently still returns the pinned-slab desc under GDS (PR #3420 flags this
   as open).
3. Should the device backend own eviction, or does the existing L1 eviction
   controller stay authoritative via `get_memory_usage`?

## Relationship to PR #3420

This proposal is **complementary**, not a replacement. The intent is:

1. Land Seam A (allocator polymorphism) as a small standalone refactor.
2. Land Seam B (`L1DeviceBackend` protocol) and re-point PR #3420's
   `GdsL1Backend` onto it as the reference implementation.
3. Add the CXL-pooled Maru backend as the second implementation.
