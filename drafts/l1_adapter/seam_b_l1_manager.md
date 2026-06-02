# Seam B — `L1Manager` before/after (draft)

This shows how `lmcache/v1/distributed/l1_manager.py` changes to depend on the
`L1DeviceBackend` protocol (`l1_device_backend.py`) instead of the concrete
`GdsL1Backend` that PR #3420 introduces. **Logic is unchanged** — only the type
and the import are generalized. GDS becomes one implementation of the protocol.

The point: with this in place, adding CXL / DAX / Maru as L1 requires **zero**
further edits to `L1Manager` — they just implement `L1DeviceBackend`.

---

## Import

```diff
-from lmcache.v1.distributed.gds_l1 import GdsL1Backend
+from lmcache.v1.distributed.l1_device_backend import L1DeviceBackend
```

## Constructor

```diff
 def __init__(
     self,
     config: L1ManagerConfig,
-    gds_backend: GdsL1Backend | None = None,
+    device_backend: L1DeviceBackend | None = None,
 ):
     ...
-    self._gds_backend = gds_backend
+    self._device_backend = device_backend
```

## `reserve_read` — fill-on-miss

```diff
     entry = self._objects.get(key, None)
     if entry is None:
-        entry = self._try_gds_fill_on_miss_locked(key)
+        entry = self._try_device_fill_on_miss_locked(key)
         if entry is None:
             ret[key] = (L1Error.KEY_NOT_EXIST, None)
             continue
```

```diff
-def _try_gds_fill_on_miss_locked(self, key: ObjectKey) -> L1ObjectState | None:
+def _try_device_fill_on_miss_locked(self, key: ObjectKey) -> L1ObjectState | None:
-    if self._gds_backend is None:
+    if self._device_backend is None:
         return None
-    gds_obj = self._gds_backend.create_memory_obj_from_index(key)
-    if gds_obj is None:
+    dev_obj = self._device_backend.create_memory_obj_from_index(key)
+    if dev_obj is None:
         return None
     new_entry = L1ObjectState(
-        memory_obj=gds_obj,
+        memory_obj=dev_obj,
         write_lock=TTLLock(self._write_ttl_seconds),
         read_lock=TTLLock(self._read_ttl_seconds),
         is_temporary=False,
     )
     self._objects[key] = new_entry
     return new_entry
```

## `reserve_write` — device-anchored allocation

```diff
-    if self._gds_backend is not None:
+    if self._device_backend is not None:
         allocated_objs = [
-            self._gds_backend.create_memory_obj(key, layout_desc)
+            self._device_backend.create_memory_obj(key, layout_desc)
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
+    if self._device_backend is not None:
+        return self._device_backend.get_memory_usage()
     return self._memory_manager.get_memory_usage()
```

## `close`

```diff
     self._memory_manager.close()
-    if self._gds_backend is not None:
-        self._gds_backend.close()
+    if self._device_backend is not None:
+        self._device_backend.close()
```

## `storage_manager.py` construction site

`StorageManager` currently builds a `GdsL1Backend` from `gds_l1_config` and
passes it as `gds_backend=`. After Seam B it passes the same object as
`device_backend=` (it already satisfies `L1DeviceBackend`). A later step can
generalize the *selection* (which device backend to build) behind config, but
that is not required for this seam.

```diff
-    l1_manager = L1Manager(l1_config, gds_backend=gds_backend)
+    # gds_backend already satisfies the L1DeviceBackend protocol
+    l1_manager = L1Manager(l1_config, device_backend=gds_backend)
```
