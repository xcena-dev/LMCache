# L1 Adapter — implementation draft (NOT for posting / NOT wired in)

These files are **review drafts only**. Nothing here is imported by the package
and nothing has been pushed anywhere. They are the concrete code sketch behind
the proposal in `docs/design/v1/distributed/l1_adapters/overall.md`.

Two independent seams, each landable as its own small PR:

| File | Seam | What it shows |
|------|------|---------------|
| `seam_a_gpu_ops.py`        | A | `gpu_ops` H2D/D2H becomes a polymorphic call on the allocator; the `isinstance` chain disappears. Default + Lazy + GDS overrides shown. |
| `l1_device_backend.py`     | B | The `L1DeviceBackend` Protocol that `L1Manager` depends on instead of concrete `GdsL1Backend`. |
| `seam_b_l1_manager.md`     | B | Before/after of the `L1Manager` call sites (constructor + `reserve_read`/`reserve_write`/`get_memory_usage`/`close`). |

Mapping to current code (dev branch):
- `lmcache/v1/gpu_connector/gpu_ops.py`
- `lmcache/v1/memory_management.py` → `MemoryAllocatorInterface` (line ~829)
- `lmcache/v1/distributed/l1_manager.py`
- `lmcache/v1/distributed/gds_l1.py` (PR #3420: `GdsScratchAllocator`, `GdsL1Backend`)
