# MaruMemoryAllocator

`lmcache/v1/distributed/maru_memory_allocator.py`

CXL-backed L1 allocator for LMCache MP mode. Wraps the embedded
`CxlMemoryAdapter` (`maru_lmcache`) and surfaces a slice of its API
through `MemoryAllocatorInterface` so the rest of the MP stack
(`L1MemoryManager` / `L1Manager` / `StorageManager`) can treat maru as
an L1 backend swap.

## Why it exists

`MaruServer` already provides everything needed for a cross-instance L1
tier — pinned-page CXL allocations, an RPC-driven KV index, dup-skip
on store, and pin/unpin lifecycle. Re-implementing that as an L2
adapter would add cascade complexity (`StoreController`,
`PrefetchController`, eviction loop, async worker pool) that the PoC
does not need. By making maru an *L1* allocator and dispatching
directly to `MaruHandler` from `L1Manager`, we get:

- **1-hop GPU↔CXL data path** — `cudaMemcpy` source/dest is the
  pool-resident `MemoryObj`, no DRAM bounce.
- **Sync linear control flow** — no eventfd / worker pool / cascade
  ordering to reason about.
- **Minimum new code** — one allocator (this file) and a deep maru
  branch in `L1Manager`; `StoreController` /
  `PrefetchController` / `L1EvictionController` are bypassed.

## Lifecycle (two-phase startup)

The MP server constructs `StorageManager` (and therefore this allocator)
*before* any vLLM worker has registered its KV cache tensors — at that
point the KV shapes/dtypes/format are unknown. But
`CxlMemoryAdapter`'s pool is **pre-typed**: it pre-creates one
`TensorMemoryObj` per CXL page with a canonical
`(shape, dtype, fmt)` tuple. So the allocator separates handler
connection from pool construction:

```
LMCache MP startup (StorageManager.__init__)
  └─ MaruMemoryAllocator.__init__(config)
       - Store config only. _handler = _cxl_adapter = None.

(later) vLLM worker registers KV cache via ZMQ RPC
  └─ MPCacheEngine.register_kv_cache(...)
       └─ StorageManager.register_kv_layout(shapes, dtypes, fmt, chunk_size)
            └─ L1Manager.register_kv_layout
                 └─ L1MemoryManager.register_kv_layout
                      └─ MaruMemoryAllocator.init_layout(shapes, dtypes, fmt, chunk_size_in_tokens)
                           - Compute full_chunk_size_bytes from shapes/dtypes
                           - MaruHandler.connect(pool_size, chunk_size_bytes)
                           - CxlMemoryAdapter(handler, shapes, dtypes, fmt, chunk_size)
                             → MaruHandler replays on_region_added
                               → _build_region_pool: TensorMemoryObj per page
                           - Cache layout tuple for subsequent-register validation
```

Once `init_layout` returns, the allocator is fully usable.
`is_initialized` becomes `True`. Subsequent calls to `init_layout` with
the same layout are no-ops; mismatched layouts raise `ValueError`
(single-model constraint, see below).

`allocate` / `batched_allocate` / `get_by_location` /
`create_store_handle` raise `RuntimeError` if invoked before
`init_layout`. On the engine hot path this is unreachable —
`MPCacheEngine.register_kv_cache` always runs before `store` / `lookup`
RPCs from the same vLLM worker.

## Single-model constraint

The CXL pool is typed at the first `init_layout` call. Subsequent
registrations with different shapes/dtypes/fmt are rejected, which
means:

- **Default DRAM backends support multi-model deployments.** A single
  LMCache MP server can serve multiple vLLM instances running
  different models simultaneously; `MixedMemoryAllocator` /
  `LazyMemoryAllocator` allocate per-call with the caller's shapes.
- **Maru backend is single-model.** All registered vLLM workers must
  share the same `(shapes, dtypes, fmt, chunk_size)` tuple. The first
  worker's layout determines the pool typing; mismatched workers
  fail to register.

This is consistent with the embedded L2 `MaruBackend` (which has the
same constraint).

### TODO(maru-multi-model)

Lift this constraint by partitioning the CXL pool by layout key and
holding one `CxlMemoryAdapter` per distinct layout. Implementation
sketch:

- `MaruMemoryAllocator` keeps `dict[LayoutKey, CxlMemoryAdapter]`.
- `init_layout` adds an entry on first sight of a new layout (rather
  than rejecting), subject to a per-layout pool quota.
- `batched_allocate` / `get_by_location` dispatch by layout — caller
  passes shapes/dtypes/fmt as today; the allocator picks the adapter
  whose canonical layout matches.
- `MaruHandler` would need either a way to partition its pool across
  multiple chunk sizes or a separate handler per layout (the latter
  is simpler but multiplies socket count).

Not needed for Phase 1 PoC. Multi-model on a single MP node is rare;
the common production case is one node per model.

## Pin / refcount semantics

`MemoryObj.parent_allocator = None` for all objects returned by this
allocator. LMCache's refcount-driven free path *must not* release the
underlying CXL pages — the lifecycle is owned 100% by `MaruServer`:

- `MaruHandler.batch_store` registers the KV index entry (and
  dup-skips if the key already exists, auto-freeing the just-allocated
  page transparently).
- `MaruHandler.batch_pin` / `batch_unpin` manage read locks against
  the KV index, not the page itself.
- `MaruHandler.delete` removes the KV entry and frees the page.

Accordingly, `free()` and `batched_free()` on this allocator are
no-ops. `close()` is best-effort and tolerates calls before
`init_layout` (handler/adapter still `None`).

## Where the boundary sits

What lives in this module:
- `MaruL1Config` dataclass — layout-independent params known at
  StorageManager construction time (`server_url`, `pool_size_bytes`,
  `instance_id`, and `MaruHandler` socket tunables).
- `MaruMemoryAllocator` — the allocator class itself.

What does *not* live here:
- `MaruHandler` RPC dispatch (in `L1Manager._maru_*` helpers — the
  allocator only exposes `handler` for those helpers to grab).
- `register_kv_cache` flow wiring (in `MPCacheEngine`).
- Controller bypass (in `StorageManager.__init__` maru branch).

## Phase 2 evolution

Phase 1 (described above) gives 1-hop MP + cross-node sharing baseline.
Phase 2 (out of scope) would slim the `L1Manager` maru branch by
introducing a `MaruL2Adapter` that reuses the standard
`L2AdapterInterface` cascade, an async RPC path, listener-pattern
observability, and the multi-adapter cascade (Maru + NIXL).
`MaruMemoryAllocator` and its allocator-interface surface are
preserved verbatim across the transition.

## See also

- `lmcache/v1/distributed/l1_manager.py` — the dispatch site
  (`_is_maru_backend`, `_maru_handler` property, `_maru_*` helpers).
- `lmcache/v1/distributed/storage_manager.py` — controller / L2
  adapter bypass.
- `lmcache/v1/multiprocess/server.py` — `register_kv_cache` invokes
  `register_kv_layout`; `store` threads `memory_objs` through
  `finish_write`.
- `lmcache/v1/storage_backend/maru_backend.py` — the embedded L2
  variant for the single-process `LMCacheEngine` path (same
  `CxlMemoryAdapter`, different wiring).
