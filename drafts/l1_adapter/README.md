# Pluggable L1 backend — implementation draft (review only, NOT wired in)

These files are **review drafts**. Nothing here is imported by the package and
nothing is pushed upstream. They are the concrete code behind
`docs/design/v1/distributed/l1_adapters/overall.md`.

This is **one cohesive change** ("make L1 backends pluggable"), landed as a
single PR — not split per integration point. A new L1 backend implements one
interface; the core (`gpu_ops`, `L1Manager`) stops hard-coding any device.

L1 ≡ GPU-DMA-able memory. The core needs exactly two things from any L1 backend:

| File | Integration point | What it shows |
|------|-------------------|---------------|
| `gpu_ops_after.py`         | the DMA itself  | `dma_to_gpu` / `dma_from_gpu` on the allocator; `gpu_ops` drops its `isinstance` chain. Default + Lazy + GDS overrides. |
| `l1_backend.py`            | alloc/lookup/usage | the `L1Backend` Protocol that `L1Manager` depends on instead of concrete `GdsL1Backend`. |
| `l1_manager_before_after.md` | alloc/lookup/usage | before/after of the `L1Manager` call sites. |

Before / after GDS:
- **Before** (PR #3420): GDS imported into `gpu_ops` + `L1Manager`, branched on directly.
- **After** (this draft): GDS becomes the *first* `L1Backend` implementation; its
  `GdsScratchAllocator` overrides `dma_to_gpu`/`dma_from_gpu`. No core edits to
  add the next backend (CXL / DAX / Maru).

Mapping to current code (dev branch):
- `lmcache/v1/gpu_connector/gpu_ops.py`
- `lmcache/v1/memory_management.py` → `MemoryAllocatorInterface` (line ~829)
- `lmcache/v1/distributed/l1_manager.py`
- `lmcache/v1/distributed/gds_l1.py` (PR #3420: `GdsScratchAllocator`, `GdsL1Backend`)
