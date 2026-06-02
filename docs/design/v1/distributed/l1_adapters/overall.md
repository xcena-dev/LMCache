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

> **This level is already multi-L1-safe by construction — and that is all it is.**
> Dispatch keys off `memory_obj.parent()`, so objects from *different* L1 backends
> can be interleaved in one transfer batch and each still routes to its own
> medium. `gpu_ops` never decides *which* L1 — that is predetermined upstream
> (by `reserve_write` / the retrieve loop) when the `MemoryObj` is created.
> Making `gpu_ops` polymorphic therefore changes **extensibility / layering only,
> not cardinality**: it removes the per-device `isinstance` edit and the
> layering inversion (the diff imports `GdsScratchAllocator` *into* `gpu_ops`),
> but it adds no multi-L1 capability because none was missing here. All multi-L1
> *orchestration* lives strictly above this level (see "Multi-L1" below).

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

## Scope: one backend now, multi-L1 later

This proposal deliberately supports **0 or 1** L1 backend — a device *swap* for
the pinned slab — reflected by the singular `backend: L1Backend | None`. That
matches how GDS actually behaves in PR #3420: it is a **replacement** for the
CPU-pinned L1 (one `_objects` index, a single `gds_backend` chosen *either/or*
against `_memory_manager`), not a tier that coexists with CPU L1.

The #3262 vision of **multiple L1s at once** (Local L1 = DRAM/DAX, Shared L1 =
CXL-pool, plus GDS) is **explicitly out of scope here.** Four kinds of logic
exist *only* when more than one L1 coexists, and none of them have — or should
have — a home in this single-backend proposal:

| Multi-L1 concern | Has no home here | 1:1 precedent already in the L2 layer |
|---|---|---|
| (a) write-target selection (which L1 gets a chunk?) | `reserve_write` is a binary `if backend is not None` | `StorePolicy.select_store_targets(keys, adapters)` |
| (b) cross-L1 read lookup order | fill-on-miss probes exactly one backend | `PrefetchPolicy.select_load_plan` (lowest-index adapter that has the key) |
| (c) promotion / demotion between L1 tiers | does not exist at any level | *(no L2 analogue — genuinely new; L2 only has vertical L2→L1 load)* |
| (d) cross-L1 capacity accounting / eviction | `get_memory_usage` returns one scalar tuple | `L2EvictionController` over `list[L2AdapterEvictionState]` |

Stating this boundary explicitly turns a silent gap into a bounded scope: the
per-backend surface (`L1Backend` + `dma_to_gpu`/`dma_from_gpu`) is the right,
coexistence-safe contract — it is the direct analogue of `L2AdapterInterface`,
which likewise holds *none* of (a)–(d). What is missing for true multi-L1 is the
orchestration **layer above** it, not anything inside the backend.

## Multi-L1: lift the L2 template (forward design, not this PR)

The codebase already solved "multiple coexisting backends" for L2, and the L1
answer should **lift that template wholesale rather than reinvent it.** There is
a structural asymmetry to repay: `StorageManager` holds a scalar `_l1_manager`
but a `list[L2AdapterInterface]` (`storage_manager.py:55` vs `:70`). L2 places
*every* multi-backend decision **one level above** the per-backend interface —
exactly where multi-L1 orchestration must sit too. When genuine multi-L1 lands,
it introduces, **above `L1Manager`** (in `StorageManager` or a new L1
coordinator) — **never inside `L1Manager` and never inside an `L1Backend`**:

1. **`backends: list[L1Backend]` with positional index identity** (mirror
   `AdapterDescriptor`) + a `create_l1_backend` registry — replacing
   `backend: L1Backend | None`.
2. **An `L1StorePolicy.select_l1_target(...)`** analogous to
   `StorePolicy.select_store_targets` — where `reserve_write` does its either/or today.
3. **An L1 lookup-order / fill-on-miss policy** analogous to
   `PrefetchPolicy.select_load_plan` — probe resident L1 tiers in a defined
   precedence (fastest / lowest-index wins) and resolve overlap.
4. **A unified L1 eviction controller** over per-tier eviction states (one
   `EvictionPolicy` + watermark + usage per backend), mirroring
   `L2EvictionController` + `L2AdapterEvictionState` — replacing the single
   `get_memory_usage` tuple.

The one multi-L1 concern that does **not** live in the coordinator is physical:
the **single `tmp_gpu_buffer_` per `GPUCacheContext`** (`gpu_context.py:142-146`).
PR #3420 cuFile-registers *every slot* of it, making the staging buffer
GDS-private. GDS needs cuFile-registered VRAM (4 KiB-aligned, ≤16 MiB regions);
plain DRAM/CXL need unregistered buffers — two L1s with different registration
disciplines **cannot share one buffer**. Multi-L1 therefore forces
`GPUCacheContext` to hold a **buffer-per-backend + allocator-per-backend** set,
and the store/retrieve loops to pick the slot whose registration matches each
chunk's target backend. This is a `GPUCacheContext`-level change, not a policy one.

See `docs/design/v1/distributed/l2_adapters/` (`store_policy`, `prefetch_policy`,
`l2_eviction`) for the prescribed pattern.

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

## Decisions & open questions (for #3262 / PR #3420)

1. **Multiple backends at once — decided: out of scope here, follow the L2
   template later.** `backend: L1Backend | None` is an *intentional*
   single-backend interim (a device swap), not a generalization to N. Genuine
   multi-L1 (#3262 Local-L1 vs Shared-L1) **will** require coexisting backends,
   and when it arrives it **must** adopt the list+index+policy+unified-controller
   shape of the L2 layer (see "Multi-L1" above) — **not** more
   `if backend is not None` branches inside `L1Manager`, which is the
   non-scaling direction PR #3420 takes (binary, concrete-typed, copy-per-device).
2. **Staging buffer under multi-L1 — recorded blocker.** Today `GPUCacheContext`
   holds a single `tmp_gpu_buffer_`; GDS cuFile-registers every slot, making it
   GDS-private. Multi-L1 needs a buffer-per-backend set at the `GPUCacheContext`
   level (see "Multi-L1" above). NIXL co-tenancy with a backend-registered VRAM
   region is the same family of problem; `get_l1_memory_desc` still returns the
   pinned-slab desc under GDS today (PR #3420 flags this) and stays unresolved
   until the per-backend buffer model lands.
3. **Eviction ownership — resolved via the L2 precedent: the controller stays
   authoritative.** As in L2 (`L2EvictionController` owns the loop; the adapter
   only reports usage and executes `delete` actions), the `L1Backend` protocol
   reports `get_memory_usage` and the (future, unified) L1 eviction controller
   decides and drives eviction. The protocol surface stays free of eviction
   policy.
