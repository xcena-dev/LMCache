# Pluggable L1 Backends (proposal / draft)

> Status: **DRAFT for discussion** — context: issue #3262 (Distributed MP mode
> RFC) and PR #3589 (merged GDS L1 slab-file tier, Shaoting-Feng, commit
> `4bbfd11b`). PR #3589 **supersedes** the earlier #3420 "GDS L1 Draft", which
> remains an open, never-merged draft; references below to "#3420 as-is" mean
> that superseded draft, not what shipped.
>
> Re-verified against merged `dev` (#3589 GDS L1, #3584 DAX L1) on 2026-06-18.
>
> Goal: make **adding a new L1 backend** a self-contained change — implement one
> interface, register it — instead of cracking open `L1Manager`, `gpu_ops`, and
> the staging-buffer layer and threading device-specific dispatch through the
> core. Merged `dev` already has a *medium-level* abstract seam
> (`L1ManagerProtocol`); the delta this proposal argues for is the
> **control-plane / multi-L1 orchestration layer** above it, plus unifying the
> **DAX-vs-GDS plug-in-level asymmetry** that the two merged tiers introduced.

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

## The two integration points

A new L1 backend touches the core in two places — the DMA dispatch in `gpu_ops`
and the allocation/usage surface exposed to `L1Manager`. In merged `dev` the
*second* of these is already abstract (`L1ManagerProtocol`); the *first* still
keys on concrete object types. They are the two faces of one "L1 backend"
concept, and this proposal's interest is in tightening both and adding the layer
above them.

### 1. The DMA itself — `gpu_ops` dispatch

This is the heart of L1. In merged `dev`, `gpu_ops.py` dispatches on the
**memory-object type**, with two branches:

```python
# merged dev (gpu_ops.py): dispatch keys on the MemoryObj type
def lmcache_memcpy_async_h2d(memory_obj, gpu_buffer):
    if isinstance(memory_obj, GDSMemoryObject):
        get_gds_context().transfer_async(memory_obj, gpu_buffer, SlabDirection.READ)
        return
    # ... size checks ...
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        lmc_ops.lmcache_memcpy_async(... H2D ...)   # pinned CPU DRAM
    else:
        gpu_buffer.view(torch.uint8).copy_(...)      # default cudaMemcpy
```

So GDS already routes through its own `GDSContext.transfer_async`, and the CPU
path is selected by `isinstance(memory_obj.parent(), LazyMemoryAllocator)`.
There is no `GdsScratchAllocator` import here — that name belongs to the
superseded #3420 draft and does **not** exist in `dev`.

A *forward* refinement (this proposal, not in `dev`) is to make the transfer a
method the medium owns, so `gpu_ops` stops growing a branch per new medium:

```python
# PROPOSAL (does not exist in dev): the medium owns its DMA
class MemoryAllocatorInterface:
    def dma_to_gpu(self, memory_obj, gpu_buffer) -> None:
        """DMA this object's bytes INTO a GPU buffer (load / H2D).
        Default = cudaMemcpyAsync from the host tensor. A non-DRAM L1
        backend overrides this with its native DMA (cuFile, CXL, ...)."""
    def dma_from_gpu(self, gpu_buffer, memory_obj) -> None:
        """DMA a GPU buffer's bytes INTO this object (evict / D2H)."""
```

so `gpu_ops` would stop knowing any concrete type:

```python
# PROPOSAL
def lmcache_memcpy_async_h2d(memory_obj, gpu_buffer):
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().dma_to_gpu(memory_obj, gpu_buffer)
```

The CPU path and the GDS path each keep their existing transfer — byte-for-byte
identical behavior, just behind a polymorphic seam instead of two `isinstance`
branches. `dma_to_gpu` / `dma_from_gpu` are **forward-proposal names only**;
they are not present in merged `dev`.

> **This level is already multi-L1-safe by construction — and that is all it is.**
> Dispatch keys off the object's identity (`isinstance(memory_obj, ...)` /
> `memory_obj.parent()`), so objects from *different* L1 media can be interleaved
> in one transfer batch and each still routes to its own medium. `gpu_ops` never
> decides *which* L1 — that is predetermined upstream (by `reserve_write` / the
> retrieve loop) when the `MemoryObj` is created. Making `gpu_ops` polymorphic
> therefore changes **extensibility / layering only, not cardinality**: it removes
> the per-medium `isinstance` edit, but it adds no multi-L1 capability because
> none was missing here. All multi-L1 *orchestration* lives strictly above this
> level (see "Multi-L1" below).

### 2. Allocation, lookup, usage — `L1ManagerProtocol` (already in dev)

A non-DRAM L1 backend also owns *where* its bytes live and *what* is resident.
Merged `dev` **already** expresses this as an abstract protocol that `L1Manager`
depends on — `L1ManagerProtocol`
(`memory_manager/l1_manager_protocol.py`):

```python
# merged dev: the abstract seam already exists
class L1ManagerProtocol(Protocol):
    def allocate(self, ...) -> tuple[L1Error, list[MemoryObj]]: ...
    def free(self, mem_objs: list[MemoryObj]) -> L1Error: ...
    def get_memory_usage(self) -> tuple[int, int]: ...        # feeds eviction
    def get_l1_memory_desc(self) -> Optional[L1MemoryDesc]: ...
    def close(self) -> None: ...
    def memcheck(self) -> bool: ...
```

`L1Manager` holds `self._memory_manager: L1ManagerProtocol` — **no concrete
import threaded into its method bodies**. It selects the concrete implementation
once, at construction, **either/or** on `config.gds_l1_config`
(`l1_manager.py:194-199`):

```python
# merged dev (l1_manager.py:194-199)
self._memory_manager: L1ManagerProtocol
if config.gds_l1_config is not None:
    self._memory_manager = GDSL1MemoryManager(config.gds_l1_config)
else:
    self._memory_manager = L1MemoryManager(config.memory_config)
```

`reserve_read` / `reserve_write` then delegate straight to
`self._memory_manager.allocate(...)` / `.free(...)` — no per-device `isinstance`
branches, no `if self._gds_backend is not None:` slot. (`GdsL1Backend` and
`gds_backend` are #3420-draft names and do **not** exist in `dev`.)

> Note the two faces correspond to two collaborating objects. The
> **manager-facing object** is now abstract via `L1ManagerProtocol`
> (`GDSL1MemoryManager` vs `L1MemoryManager`). The **DMA itself** is *not* yet
> abstracted at the same level — `gpu_ops` still keys on the concrete
> `GDSMemoryObject` type. The proposal's integration-point-1 work closes that
> remaining gap; integration point 2 is largely done in `dev`.

## Before / after GDS

The whole proposal is best read as "what GDS looks like before vs. after." The
"before" baseline below is the **merged #3589** wiring, not the superseded #3420
draft.

**Before (merged #3589 as shipped):** GDS is a self-contained tier behind one
abstract seam, with one remaining concrete dispatch.
- `gpu_ops.py` dispatches on `isinstance(memory_obj, GDSMemoryObject)` →
  `get_gds_context().transfer_async(...)`. This is the one place still keyed on a
  concrete GDS type.
- `l1_manager.py` depends on `L1ManagerProtocol` abstractly and selects
  `GDSL1MemoryManager` XOR `L1MemoryManager` once at construction
  (`l1_manager.py:194-199`). There are **no** per-device `isinstance` branches in
  `reserve_read` / `reserve_write` / the rest — they delegate to
  `self._memory_manager`.
- GDS has **no separate allocator class**: `GDSL1MemoryManager` owns its own
  `AddressManager` and mints `GDSMemoryObject` directly.

> The "non-scaling, copy-per-device, concrete-typed `isinstance`-in-`L1Manager`"
> critique applies to the **literal superseded #3420 draft** — it is *not* how
> #3589 shipped. #3589 deliberately avoided concrete-typing `L1Manager` by
> introducing `L1ManagerProtocol`. The live, still-valid critique is narrower:
> (a) `gpu_ops` still branches on the concrete `GDSMemoryObject`, and (b) GDS and
> DAX plugged in at **different levels** (see "Two shapes" below).

**After (this proposal):** the core knows only interfaces, end to end.
- `gpu_ops.py` calls `parent().dma_to_gpu(...)` — no concrete-type branch.
- `l1_manager.py` keeps depending on `L1ManagerProtocol` (already true) and the
  *shape* of how each medium plugs in is unified (see below).
- **GDS stays the first non-DRAM `L1ManagerProtocol` implementation**
  (`GDSL1MemoryManager`); its `GDSMemoryObject` transfer moves behind
  `dma_to_gpu`/`dma_from_gpu`. Nothing about the GDS data path or perf changes.
- Adding a further medium = implement the interface, register it. **Near-zero
  core edits.**

## Two shapes today: DAX vs GDS (the asymmetry to repay)

Merged `dev` actually has **two** non-DRAM L1 media, and they plug in at
**different levels** — this is the concrete asymmetry the unification work
targets:

- **GDS (#3589):** a *separate manager* — `GDSL1MemoryManager` selected behind
  `L1ManagerProtocol`, owning its own `AddressManager` and minting
  `GDSMemoryObject`.
- **DAX (#3584, merged 2026-06-17):** *not* a separate manager. It ships as
  `DevDaxMemoryAllocator(MemoryAllocatorInterface)` used **inside** the standard
  `L1MemoryManager` (the manager builds it and checks
  `isinstance(self._allocator, DevDaxMemoryAllocator)`), with a DRAM-primary
  `MixedMemoryAllocator` and the DAX allocator as overflow. CLI: `--l1-devdax-path`.

So one new medium became a *new manager behind the protocol* and the other became
*an allocator inside the existing manager*. Both are legitimate; the cost is that
the "how do I add an L1 medium?" answer is currently **two shapes**, not one.
Unifying these (one prescribed plug-in level) is part of this proposal's delta.

## What does NOT change

- The `MemoryObj` subclassing model and the existing
  `MemoryAllocatorInterface` implementations — untouched.
- The CPU-pinned DRAM path stays the default, byte-for-byte unchanged.
- MP server / `StorageManager` data path — unaffected (already routes through
  `gpu_ops`).
- GDS throughput / cuFile P2P DMA behavior — identical; only its `gpu_ops`
  *wiring* would move behind the interface.

## Scope: one L1 medium at a time now, multi-L1 later

Merged `dev` deliberately supports **exactly one** non-DRAM L1 medium at a time —
a device *swap* for the pinned slab, reflected by the either/or selection
`self._memory_manager = GDSL1MemoryManager(...)` **XOR** `L1MemoryManager(...)`
(`l1_manager.py:194-199`). GDS is a **replacement** for the CPU-pinned L1, not a
tier that coexists with it. (DAX is likewise gated: its config inference consumes
a matching DAX L2 adapter into L1 overflow, and coexistence with
`nixl_store` / `nixl_store_dynamic` / RDMA `mooncake_store` is rejected.)

The #3262 vision of **multiple L1s at once** (Local L1 = DRAM/DAX, Shared L1 =
CXL-pool, plus GDS) is **explicitly out of scope here.** Four kinds of logic
exist *only* when more than one L1 coexists, and none of them have — or should
have — a home in the single-medium model:

| Multi-L1 concern | Has no home today | 1:1 precedent already in the L2 layer |
|---|---|---|
| (a) write-target selection (which L1 gets a chunk?) | selection is the either/or in `l1_manager.py:194-199` | `StorePolicy.select_store_targets(keys, adapters)` |
| (b) cross-L1 read lookup order | resolution probes exactly one medium | `PrefetchPolicy.select_load_plan` (lowest-index adapter that has the key) |
| (c) promotion / demotion between L1 tiers | does not exist at any level | *(no L2 analogue — genuinely new; L2 only has vertical L2→L1 load)* |
| (d) cross-L1 capacity accounting / eviction | `get_memory_usage` returns one scalar tuple | `L2EvictionController` over `list[L2AdapterEvictionState]` |

Stating this boundary explicitly turns a silent gap into a bounded scope: the
per-medium surface (`L1ManagerProtocol` + a future `dma_to_gpu`/`dma_from_gpu`)
is the right, coexistence-safe contract — it is the direct analogue of
`L2AdapterInterface`, which likewise holds *none* of (a)–(d). What is missing for
true multi-L1 is the orchestration **layer above** it, not anything inside the
per-medium implementation.

## Multi-L1: lift the L2 template (forward design, not this PR)

The codebase already solved "multiple coexisting backends" for L2, and the L1
answer should **lift that template wholesale rather than reinvent it.** There is
a structural asymmetry to repay: `StorageManager` holds a **scalar**
`self._l1_manager` (`storage_manager.py:65`) but a
`self._l2_adapters: dict[int, L2AdapterInterface]` keyed by adapter id
(`storage_manager.py:86`), passed to controllers as
`list(self._l2_adapters.values())`. L2 places *every* multi-backend decision
**one level above** the per-backend interface — exactly where multi-L1
orchestration must sit too. When genuine multi-L1 lands, it introduces, **above
`L1Manager`** (in `StorageManager` or a new L1 coordinator) — **never inside
`L1Manager` and never inside a per-medium `L1ManagerProtocol` implementation**:

1. **`backends: list[...]` with positional index identity** (mirror
   `AdapterDescriptor`) + a `create_l1_backend` registry — replacing the
   either/or `GDSL1MemoryManager` XOR `L1MemoryManager` selection.
2. **An `L1StorePolicy.select_l1_target(...)`** analogous to
   `StorePolicy.select_store_targets` — where the either/or selection sits today.
3. **An L1 lookup-order / fill-on-miss policy** analogous to
   `PrefetchPolicy.select_load_plan` — probe resident L1 tiers in a defined
   precedence (fastest / lowest-index wins) and resolve overlap.
4. **A unified L1 eviction controller** over per-tier eviction states (one
   `EvictionPolicy` + watermark + usage per backend), mirroring
   `L2EvictionController` + `L2AdapterEvictionState` — replacing the single
   `get_memory_usage` tuple.

The one multi-L1 concern that does **not** live in the coordinator is physical:
the **GPU staging buffer**. Today the GDS tier registers its staging buffer via
`GDSContext.register_gpu_buffer(buffer)` (`gpu_connector/gds_context.py`), which
cuFile-registers the buffer in ≤16 MiB regions (`_MAX_CUFILE_REGION`,
4 KiB-aligned via `_CUFILE_ALIGNMENT`). That makes the registered VRAM
GDS-private: GDS needs cuFile-registered, 4 KiB-aligned, ≤16 MiB regions; plain
DRAM/CXL need unregistered buffers — two L1s with different registration
disciplines **cannot share one buffer**. Multi-L1 therefore forces the gpu
connector layer (`gpu_connector/gpu_connectors.py`; there is no `gpu_context.py`
in `dev`) to hold a **buffer-per-backend + registration-per-backend** set, and
the store/retrieve loops to pick the slot whose registration matches each chunk's
target medium. This is a connector-layer change, not a policy one.

See `docs/design/v1/distributed/l2_adapters/` (`store_policy`, `prefetch_policy`,
`l2_eviction`) for the prescribed pattern.

## Rollout

- **One PR**: introduce `dma_to_gpu`/`dma_from_gpu` on the allocator interface,
  switch `gpu_ops` off the concrete `GDSMemoryObject` branch onto them, and
  re-point GDS through that seam. It is one cohesive "make the L1 DMA dispatch
  pluggable" change; the allocator refactor is behavior-preserving on its own.
  Since GDS L1 (#3589) is already merged, this PR refactors the merged GDS onto
  the interface rather than racing it.
- **Follow-up PR(s)**: unify the DAX-vs-GDS plug-in shape (one prescribed level
  for adding an L1 medium), then the multi-L1 coordinator above `L1Manager`.
- **A further backend**: the XCENA CXL-pooled **Maru** L1 medium as an additional
  implementation behind the unified seam — a genuinely separate device, so a
  separate PR. (DAX, #3584, is already the second non-DRAM L1 medium, shipped as
  `DevDaxMemoryAllocator` inside `L1MemoryManager`; Maru would be a *further* one.)

> The scaffolding under `drafts/l1_adapter/` (`gpu_ops_after.py`, `l1_backend.py`,
> `l1_manager_before_after.md`, `README.md`) was written against the **#3420
> draft** and is now **stale** — its `L1Backend` / `dma_to_gpu` sketches predate
> the merged `L1ManagerProtocol` / `GDSL1MemoryManager` surface. Treat those
> files as historical sketches; this doc, not that scaffolding, is the current
> reference. (They are not rewritten here.)

## Decisions & open questions (for #3262 / #3589 / #3584)

1. **Multiple backends at once — decided: out of scope here, follow the L2
   template later.** The single-medium either/or in `l1_manager.py:194-199` is an
   *intentional* interim (a device swap), not a generalization to N. Genuine
   multi-L1 (#3262 Local-L1 vs Shared-L1) **will** require coexisting backends,
   and when it arrives it **must** adopt the list+index+policy+unified-controller
   shape of the L2 layer (see "Multi-L1" above) — sitting **above** `L1Manager`,
   not as more concrete branches inside it. (The literal #3420 draft took the
   binary, concrete-typed, copy-per-device direction; #3589 already avoided that
   via `L1ManagerProtocol`.)
2. **Staging buffer under multi-L1 — recorded blocker.** The GDS tier
   cuFile-registers its staging buffer via `GDSContext.register_gpu_buffer`
   (≤16 MiB, 4 KiB-aligned), making it GDS-private. Multi-L1 needs a
   buffer-per-backend set at the gpu connector layer
   (`gpu_connector/gpu_connectors.py`; see "Multi-L1" above). NIXL co-tenancy with
   a backend-registered VRAM region is the same family of problem. Note that
   under merged GDS, `GDSL1MemoryManager.get_l1_memory_desc()` returns **`None`**
   (`gds_l1_memory_manager.py:114`) — there is no exported pinned-slab descriptor
   for a NIXL peer to register against — so NIXL co-tenancy with the GDS tier is
   unresolved until a per-backend descriptor/buffer model lands.
3. **Eviction ownership — resolved via the L2 precedent: the controller stays
   authoritative.** As in L2 (`L2EvictionController` owns the loop; the adapter
   only reports usage and executes `delete` actions), `L1ManagerProtocol` reports
   `get_memory_usage` and the (current single-tier) `L1EvictionController` — and
   its future unified, multi-tier successor — decides and drives eviction. The
   protocol surface stays free of eviction policy.
4. **Durable residency / fill-on-miss — not a GDS property today.** A
   `create_memory_obj_from_index`-style "is `key` durably resident on this
   medium?" probe is **aspirational**: merged GDS clears its slab at startup and
   keeps no durable on-disk index, so it has no fill-on-miss / durably-resident
   behavior. Any cross-L1 lookup policy (concern (b) above) must treat durable
   residency as a *future* capability, not something the merged GDS tier provides.