# Pluggable L1 Backends (proposal / draft)

> Status: **DRAFT for discussion** — context: issue #3262 (Distributed MP mode
> RFC) and PR #3420 (GDS L1 Draft).
>
> Goal: make **adding a new L1 backend** a self-contained change — implement one
> interface, register it — instead of cracking open `L1Manager` and `gpu_ops`
> and threading device-specific `isinstance` / `if backend is not None` branches
> through the core, the way GDS does today.

## L1 ≡ GPU-DMA-able memory

Issue #3262 settled the definition of L1:

> **L1 = memory an xPU can access directly via DMA, without staging.**

That is the *only* property that makes something L1. It is not "CPU DRAM" — DRAM
is just the first instance. Anything the GPU can DMA to/from without a host
bounce qualifies:

| L1 medium        | the DMA it performs                                   | staging? |
|------------------|-------------------------------------------------------|----------|
| pinned CPU DRAM  | `cudaMemcpyAsync` (PCIe DMA reads pinned host direct) | no       |
| **GDS (NVMe)**   | `cuFileRead` — NVMe→VRAM **P2P DMA**, bypasses CPU     | no       |
| CXL-pool / DAX   | byte-addressable; cudaMemcpy / direct DMA             | no       |
| ~~Redis / S3~~   | **cannot** — must stage through CPU first             | yes → **L2, not L1** |

So the abstraction starts from the DMA capability: **a memory is an L1 backend
iff it can move its bytes into/out of a GPU buffer as a real DMA.** That single
operation is what the core needs from any L1 backend; everything else
(allocation bookkeeping, a durable index, usage reporting) is secondary
management that rides on top.

## The two integration points (one change, one PR)

A new L1 backend touches the core in exactly two places. They are **not two
separate features or two PRs** — they are the two faces of one "L1 backend"
interface, and land together.

### 1. The DMA itself — `dma_to_gpu` / `dma_from_gpu`

This is the heart of L1. Today `gpu_ops.py` hard-codes the transfer per medium:

```python
# today (dev): a device-specific isinstance chain
if isinstance(parent, LazyMemoryAllocator):
    lmc_ops.lmcache_memcpy_async(... H2D ...)
# PR #3420 adds a THIRD branch here:
elif isinstance(parent, GdsScratchAllocator):
    parent.cufile_read_into(memory_obj, gpu_buffer)
else:
    gpu_buffer.copy_(src_tensor...)   # default cudaMemcpy
```

Instead, the DMA becomes a method the backend's allocator owns — because the
thing that holds the bytes is the thing that knows how to DMA them:

```python
class MemoryAllocatorInterface:
    def dma_to_gpu(self, memory_obj, gpu_buffer) -> None:
        """DMA this object's bytes INTO a GPU buffer (load / H2D).
        Default = cudaMemcpyAsync from the host tensor. A non-DRAM L1
        backend overrides this with its native DMA (cuFile, CXL, ...)."""
    def dma_from_gpu(self, gpu_buffer, memory_obj) -> None:
        """DMA a GPU buffer's bytes INTO this object (evict / D2H)."""
```

and `gpu_ops` stops knowing any concrete type:

```python
def lmcache_memcpy_async_h2d(memory_obj, gpu_buffer):
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().dma_to_gpu(memory_obj, gpu_buffer)
```

`LazyMemoryAllocator` and `GdsScratchAllocator` each override with their existing
transfer — byte-for-byte identical behavior, just behind the interface.

### 2. Allocation, lookup, usage — the `L1Backend` protocol

A non-DRAM L1 backend also owns *where* its bytes live and *what* is resident.
Today PR #3420 expresses this by importing `GdsL1Backend` into `L1Manager` and
adding `if self._gds_backend is not None:` branches in five methods. Extract that
into a protocol `L1Manager` depends on abstractly:

```python
class L1Backend(Protocol):
    def create_memory_obj(self, key, layout) -> MemoryObj: ...
    # mint a backend-anchored object on reserve_write

    def create_memory_obj_from_index(self, key) -> MemoryObj | None: ...
    # fill-on-miss: is `key` durably resident on this backend?

    def get_memory_usage(self) -> tuple[int, int]: ...   # feeds eviction
    def close(self) -> None: ...
```

`L1Manager` holds `backend: L1Backend | None` (no concrete import); the existing
branches become `if self._backend is not None:` — same logic, generalized type.

> Note the two faces correspond to the two collaborating objects a backend
> already ships: the **DMA-able allocator** (`*ScratchAllocator`, parent of the
> memory objects → integration point 1) and the **manager-facing backend**
> (`*L1Backend` → integration point 2). The interface just names the contract
> the core relies on, so the core no longer hard-codes either one.

## Before / after GDS

The whole proposal is best read as "what GDS looks like before vs. after."

**Before (PR #3420 as-is):** GDS is wired into the core directly.
- `gpu_ops.py` imports `GdsScratchAllocator`, adds an `isinstance` branch.
- `l1_manager.py` imports `GdsL1Backend`, takes it as a concrete ctor arg, and
  branches on it in `reserve_read`, `reserve_write`, `get_memory_usage`,
  `get_l1_memory_desc`, `close`.
- Adding CXL/DAX/Maru later = copy all of that again for each device.

**After (this proposal):** the core knows only two interfaces.
- `gpu_ops.py` calls `parent().dma_to_gpu(...)` — no device import, no branch.
- `l1_manager.py` depends on `L1Backend` — no device import, branches are
  device-agnostic.
- **GDS becomes the first `L1Backend` implementation** (its `GdsScratchAllocator`
  overrides `dma_to_gpu`/`dma_from_gpu`; its `GdsL1Backend` implements
  `L1Backend`). Nothing about the GDS data path or perf changes.
- Adding CXL/DAX/Maru later = implement the interface, register it. **Zero core
  edits.**

## What does NOT change

- The `MemoryObj` subclassing model and the ~13 existing
  `MemoryAllocatorInterface` implementations — untouched.
- The CPU-pinned DRAM path stays the default, byte-for-byte unchanged.
- MP server / `StorageManager` data path — unaffected (already routes through
  `gpu_ops`).
- GDS throughput / cuFile P2P DMA behavior — identical; only its *wiring* moves
  behind the interface.

## Rollout

- **One PR**: introduce `dma_to_gpu`/`dma_from_gpu` on the allocator interface +
  the `L1Backend` protocol, switch `gpu_ops`/`L1Manager` onto them, and re-point
  GDS as the first implementation. It is one cohesive "make L1 backends
  pluggable" change; splitting it finer adds review overhead without isolating
  risk (the allocator refactor is behavior-preserving on its own).
  - Ordering vs. PR #3420: if #3420 merges first, this PR refactors the merged
    GDS onto the interface; if not, GDS builds on the interface from the start.
    Either way the end state is identical.
- **Follow-up PR**: the XCENA CXL-pooled **Maru** L1 backend as the second
  `L1Backend` implementation — a genuinely separate device, so a separate PR.

## Open questions (for #3262 / PR #3420)

1. Can `L1Manager` hold **more than one** backend at once (local DRAM + shared
   CXL pool), or is one-at-a-time enough for now? (#3262's "Local L1 vs Shared
   L1" split implies eventually >1.)
2. NIXL co-tenancy with a backend-registered VRAM region — `get_l1_memory_desc`
   still returns the pinned-slab desc under GDS today (PR #3420 flags this).
3. Does the backend own eviction, or does the existing L1 eviction controller
   stay authoritative via `get_memory_usage`?
