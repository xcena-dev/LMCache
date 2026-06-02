# `L1Manager` before / after (draft)

How `lmcache/v1/distributed/l1_manager.py` changes to depend on the `L1Backend`
protocol (`l1_backend.py`) instead of the concrete `GdsL1Backend` that PR #3420
introduces. **Logic is unchanged** — only the type and the import are
generalized. GDS becomes one implementation of the protocol.

The point: with this in place, adding CXL / DAX / Maru as an L1 backend requires
**zero** further edits to `L1Manager` — they just implement `L1Backend`.

---

## Import

```diff
-from lmcache.v1.distributed.gds_l1 import GdsL1Backend
+from lmcache.v1.distributed.l1_backend import L1Backend
```

## Constructor

```diff
 def __init__(
     self,
     config: L1ManagerConfig,
-    gds_backend: GdsL1Backend | None = None,
+    backend: L1Backend | None = None,
 ):
     ...
-    self._gds_backend = gds_backend
+    self._backend = backend
```

## `reserve_read` — fill-on-miss

```diff
     entry = self._objects.get(key, None)
     if entry is None:
-        entry = self._try_gds_fill_on_miss_locked(key)
+        entry = self._try_backend_fill_on_miss_locked(key)
         if entry is None:
             ret[key] = (L1Error.KEY_NOT_EXIST, None)
             continue
```

```diff
-def _try_gds_fill_on_miss_locked(self, key: ObjectKey) -> L1ObjectState | None:
+def _try_backend_fill_on_miss_locked(self, key: ObjectKey) -> L1ObjectState | None:
-    if self._gds_backend is None:
+    if self._backend is None:
         return None
-    gds_obj = self._gds_backend.create_memory_obj_from_index(key)
-    if gds_obj is None:
+    obj = self._backend.create_memory_obj_from_index(key)
+    if obj is None:
         return None
     new_entry = L1ObjectState(
-        memory_obj=gds_obj,
+        memory_obj=obj,
         write_lock=TTLLock(self._write_ttl_seconds),
         read_lock=TTLLock(self._read_ttl_seconds),
         is_temporary=False,
     )
     self._objects[key] = new_entry
     return new_entry
```

## `reserve_write` — backend-anchored allocation

```diff
-    if self._gds_backend is not None:
+    if self._backend is not None:
         allocated_objs = [
-            self._gds_backend.create_memory_obj(key, layout_desc)
+            self._backend.create_memory_obj(key, layout_desc)
             for key, _ in need_to_allocate
         ]
         err = L1Error.SUCCESS
     else:
         err, allocated_objs = self._memory_manager.allocate(
             layout_desc, len(need_to_allocate)
         )
```

## `get_memory_usage`

```diff
-    if self._gds_backend is not None:
-        return self._gds_backend.get_memory_usage()
+    if self._backend is not None:
+        return self._backend.get_memory_usage()
     return self._memory_manager.get_memory_usage()
```

## `close`

```diff
     self._memory_manager.close()
-    if self._gds_backend is not None:
-        self._gds_backend.close()
+    if self._backend is not None:
+        self._backend.close()
```

## `storage_manager.py` construction site

`StorageManager` currently builds a `GdsL1Backend` from `gds_l1_config` and
passes it as `gds_backend=`. After this change it passes the same object as
`backend=` (it already satisfies `L1Backend`). Generalizing the *selection*
(which backend to build, from config) is a later step, not required here.

```diff
-    l1_manager = L1Manager(l1_config, gds_backend=gds_backend)
+    # the GDS backend already satisfies the L1Backend protocol
+    l1_manager = L1Manager(l1_config, backend=gds_backend)
```
