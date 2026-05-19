# SPDX-License-Identifier: Apache-2.0

"""Maru-backed MP L2 adapter.

Stores L1 (DRAM) ``MemoryObj`` payloads in a CXL pool managed by
``MaruServer``. The adapter keeps L1 as the default DRAM allocator
and uses ``MaruHandler`` directly for the L2 tier — engine hot path
is ``GPU ↔ DRAM (cudaMemcpy) ↔ CXL (DRAM↔CXL memcpy via adapter
worker)``.

Registered under ``--l2-adapter '{"type":"maru",...}'`` via
``register_l2_adapter_factory``.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional
import ctypes
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


def _parse_positive_int(value, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _object_key_to_string(key: ObjectKey) -> str:
    """``ObjectKey`` → MaruServer-stable string form.

    The format mirrors the L1 dispatcher's
    :func:`lmcache.v1.distributed.maru_l1_dispatch.object_key_to_string`
    so KV entries are interoperable between the two paths.
    """
    base = f"{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _memoryview_addr(mv: memoryview) -> int:
    """Return the raw memory address of ``mv``'s first byte.

    Uses :func:`numpy.frombuffer` for a zero-copy uint8 view; the
    numpy array exposes the underlying pointer via ``ctypes.data``.
    Works for any writable / readable memoryview backed by contiguous
    memory (which MaruHandler's ``AllocHandle.buf`` and
    ``MemoryInfo.view`` both are).
    """
    return int(np.frombuffer(mv, dtype=np.uint8).ctypes.data)


class MaruL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the maru L2 adapter.

    The adapter connects to a ``MaruServer`` at ``server_url`` and
    requests a CXL pool sized at ``pool_size_gb``. ``chunk_size_bytes``
    sets the MaruServer page size (must match the LMCache full-chunk
    byte budget for the running model) and is required at construction
    because ``MaruHandler.connect()`` needs it eagerly.

    TODO(maru-l2-lazy-layout): mirror the L1 path's two-phase
    ``init_layout`` so this adapter can defer ``MaruHandler.connect()``
    until the first store and derive ``chunk_size_bytes`` from the
    inbound ``MemoryObj`` metadata, removing the need for the user to
    spell it out in CLI / yaml.
    """

    def __init__(
        self,
        *,
        server_url: str,
        pool_size_gb: float,
        chunk_size_bytes: int,
        instance_id: Optional[str] = None,
        num_store_workers: int = 1,
        num_lookup_workers: int = 1,
        num_load_workers: int = min(4, os.cpu_count() or 1),
        timeout_ms: int = 5000,
        use_async_rpc: bool = True,
        max_inflight: int = 64,
        eager_map: bool = True,
    ) -> None:
        """Build a validated config.

        Args:
            server_url: MaruServer endpoint (``maru://host:port`` or
                ``tcp://host:port``; the former is rewritten internally).
            pool_size_gb: CXL pool quota requested from MaruServer (GB).
            chunk_size_bytes: MaruServer page / chunk size. Must equal
                the running model's full KV chunk byte size.
            instance_id: Stable client identifier reported to MaruServer
                (UUID auto-generated if ``None``).
            num_store_workers: Worker threads for store tasks.
            num_lookup_workers: Worker threads for lookup-and-lock tasks.
            num_load_workers: Worker threads for load tasks.
            timeout_ms: Socket timeout for MaruHandler RPCs.
            use_async_rpc: Whether to use the DEALER-ROUTER async RPC
                client (matches MaruHandler default).
            max_inflight: Max concurrent in-flight async RPCs.
            eager_map: Whether MaruHandler should pre-map all shared
                regions on connect.
        """
        self.server_url = server_url
        self.pool_size_gb = pool_size_gb
        self.chunk_size_bytes = chunk_size_bytes
        self.instance_id = instance_id
        self.num_store_workers = num_store_workers
        self.num_lookup_workers = num_lookup_workers
        self.num_load_workers = num_load_workers
        self.timeout_ms = timeout_ms
        self.use_async_rpc = use_async_rpc
        self.max_inflight = max_inflight
        self.eager_map = eager_map

    @classmethod
    def from_dict(cls, d: dict) -> "MaruL2AdapterConfig":
        """Build the config from a ``--l2-adapter`` JSON object.

        Args:
            d: Parsed CLI JSON.

        Returns:
            A validated ``MaruL2AdapterConfig``.

        Raises:
            ValueError: If a required field is missing or any numeric
                field fails its positivity check.
        """
        server_url = d.get("server_url")
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("server_url must be a non-empty string")

        pool_size_gb = d.get("pool_size_gb")
        if not isinstance(pool_size_gb, (int, float)) or pool_size_gb <= 0:
            raise ValueError("pool_size_gb must be a positive number")

        chunk_size_bytes = d.get("chunk_size_bytes")
        if not isinstance(chunk_size_bytes, int) or chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be a positive integer")

        instance_id = d.get("instance_id")
        if instance_id is not None and not isinstance(instance_id, str):
            raise ValueError("instance_id must be a string when provided")

        num_store_workers = _parse_positive_int(
            d.get("num_store_workers", 1), "num_store_workers"
        )
        num_lookup_workers = _parse_positive_int(
            d.get("num_lookup_workers", 1), "num_lookup_workers"
        )
        num_load_workers = _parse_positive_int(
            d.get("num_load_workers", min(4, os.cpu_count() or 1)),
            "num_load_workers",
        )
        timeout_ms = _parse_positive_int(d.get("timeout_ms", 5000), "timeout_ms")
        max_inflight = _parse_positive_int(d.get("max_inflight", 64), "max_inflight")

        use_async_rpc = bool(d.get("use_async_rpc", True))
        eager_map = bool(d.get("eager_map", True))

        return cls(
            server_url=server_url.strip(),
            pool_size_gb=float(pool_size_gb),
            chunk_size_bytes=chunk_size_bytes,
            instance_id=instance_id,
            num_store_workers=num_store_workers,
            num_lookup_workers=num_lookup_workers,
            num_load_workers=num_load_workers,
            timeout_ms=timeout_ms,
            use_async_rpc=use_async_rpc,
            max_inflight=max_inflight,
            eager_map=eager_map,
        )

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for the maru L2 adapter config."""
        return (
            "Maru L2 adapter config fields:\n"
            "- server_url (str): MaruServer endpoint, maru:// or "
            "tcp:// (required)\n"
            "- pool_size_gb (float): CXL pool size to request (required, >0)\n"
            "- chunk_size_bytes (int): MaruServer page size, must match the "
            "model's full KV chunk byte size (required, >0)\n"
            "- instance_id (str): client identifier (optional; UUID if "
            "omitted)\n"
            "- num_store_workers (int): store worker threads "
            "(optional, default 1)\n"
            "- num_lookup_workers (int): lookup worker threads "
            "(optional, default 1)\n"
            "- num_load_workers (int): load worker threads "
            "(optional, default min(4, cpu_count))\n"
            "- timeout_ms (int): RPC socket timeout (optional, default 5000)\n"
            "- use_async_rpc (bool): async DEALER-ROUTER (optional, default true)\n"
            "- max_inflight (int): concurrent in-flight RPCs "
            "(optional, default 64)\n"
            "- eager_map (bool): pre-map regions on connect "
            "(optional, default true)"
        )


class MaruL2Adapter(L2AdapterInterface):
    """MP L2 adapter that stores KV chunks in a CXL pool via MaruServer.

    Threading model:
        Three independent ``ThreadPoolExecutor`` pools serve store,
        lookup-and-lock, and load tasks. Each completion signals its
        dedicated ``EventNotifier`` so the store / prefetch controllers
        can poll completions without cross-talk. ``MaruHandler`` itself
        is thread-safe (async DEALER-ROUTER); the adapter's task
        bookkeeping is guarded by ``self._lock``.
    """

    def __init__(self, config: MaruL2AdapterConfig) -> None:
        """Connect to MaruServer and prepare worker pools / event fds.

        Args:
            config: Validated ``MaruL2AdapterConfig``.

        Raises:
            RuntimeError: If ``MaruHandler.connect()`` fails.
        """
        super().__init__(max_capacity_bytes=int(config.pool_size_gb * 1024**3))
        self._config = config

        # MaruHandler / runtime types are imported lazily so this
        # module can be imported in environments without the maru
        # runtime installed (mirroring MaruMemoryAllocator's pattern).
        self._handler: Any = self._create_handler(config)

        # Three distinct event notifiers — controllers' fd-to-adapter
        # dispatch maps require them to be unique per task type.
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Lazily-built thread pools so close() can tear them down
        # without surprising shutdown races when a task is mid-flight.
        self._store_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=config.num_store_workers,
            thread_name_prefix="maru-l2-store",
        )
        self._lookup_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=config.num_lookup_workers,
            thread_name_prefix="maru-l2-lookup",
        )
        self._load_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=config.num_load_workers,
            thread_name_prefix="maru-l2-load",
        )

        # Task bookkeeping — shape matches the DAX adapter so the store
        # / prefetch controllers see a familiar surface.
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._inflight_store_tasks = 0
        self._inflight_lookup_tasks = 0
        self._inflight_load_tasks = 0

        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_handler(config: MaruL2AdapterConfig) -> Any:
        """Build and connect a ``MaruHandler`` for this adapter.

        ``maru`` is imported lazily so the adapter module can be
        loaded without the maru runtime installed.
        """
        # Third Party
        from maru import MaruConfig, MaruHandler

        server_url = config.server_url
        if server_url.startswith("maru://"):
            server_url = "tcp://" + server_url[len("maru://") :]

        maru_config = MaruConfig(
            server_url=server_url,
            instance_id=config.instance_id,
            pool_size=int(config.pool_size_gb * 1024**3),
            chunk_size_bytes=config.chunk_size_bytes,
            auto_connect=False,
            timeout_ms=config.timeout_ms,
            use_async_rpc=config.use_async_rpc,
            max_inflight=config.max_inflight,
            eager_map=config.eager_map,
        )

        handler = MaruHandler(maru_config)
        if not handler.connect():
            raise RuntimeError(f"Failed to connect MaruHandler to {config.server_url}")
        logger.info(
            "[MaruL2Adapter] connected: server=%s instance_id=%s "
            "pool_gb=%s chunk_size_bytes=%d",
            config.server_url,
            handler.instance_id,
            config.pool_size_gb,
            config.chunk_size_bytes,
        )
        return handler

    def _get_next_task_id_locked(self) -> L2TaskId:
        """Return a fresh task id; caller must hold ``self._lock``."""
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _ensure_open_locked(self) -> None:
        """Raise if the adapter has been closed; caller must hold ``self._lock``."""
        if self._closed:
            raise RuntimeError("MaruL2Adapter has been closed")

    def _signal_eventfd(self, notifier) -> None:
        """Wake any controller waiting on this notifier.

        Failures during shutdown (eventfd already closed) are
        downgraded to debug logs — the calling worker is already on
        its way out.
        """
        try:
            notifier.notify()
        except OSError:
            logger.debug("MaruL2Adapter: eventfd notify skipped (closed)")

    # ------------------------------------------------------------------
    # Event fd accessors
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store path
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous DRAM→CXL store task.

        Args:
            keys: Object keys to register with MaruServer.
            objects: Aligned L1 (DRAM) ``MemoryObj`` instances whose
                bytes will be copied into freshly-allocated CXL pages.

        Returns:
            Task id usable with :meth:`pop_completed_store_tasks`.
        """
        if len(keys) != len(objects):
            raise ValueError(
                f"keys and objects length mismatch ({len(keys)} vs {len(objects)})"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_store_tasks += 1

        assert self._store_executor is not None
        self._store_executor.submit(self._execute_store_task, task_id, keys, objects)
        return task_id

    def _execute_store_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        """Worker entry: alloc CXL page, memcpy DRAM→CXL, batch_store."""
        success = True
        stored_sizes: list[int] = []
        try:
            handles: list[Any] = []
            for obj in objects:
                size = obj.get_size()
                handle = self._handler.alloc(size)
                # DRAM → CXL byte copy. ``obj.data_ptr`` is the
                # source (L1 DRAM); the CXL page is mapped behind
                # ``handle.buf`` (zero-copy memoryview).
                dst_addr = _memoryview_addr(handle.buf)
                ctypes.memmove(dst_addr, obj.data_ptr, size)
                handles.append(handle)
                stored_sizes.append(size)

            key_strs = [_object_key_to_string(k) for k in keys]
            results = self._handler.batch_store(key_strs, handles)
            # ``batch_store`` returns per-key flags. Treat dup-skip
            # (True from server) as success — the KV is in maru regardless.
            success = all(results)
            if not success:
                # Partial-failure aggregation isn't supported by the
                # store task contract — surface the overall flag.
                logger.warning(
                    "MaruL2Adapter: batch_store reported some failures "
                    "(task_id=%d, total=%d, ok=%d)",
                    task_id,
                    len(results),
                    sum(1 for r in results if r),
                )
        except Exception:
            logger.exception(
                "MaruL2Adapter: store task %d failed (keys=%d)",
                task_id,
                len(keys),
            )
            success = False

        with self._lock:
            self._completed_store_tasks[task_id] = success
            self._inflight_store_tasks -= 1

        if success and stored_sizes:
            self._notify_keys_stored(keys, stored_sizes)
        self._signal_eventfd(self._store_efd)

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Hand the controller every completed store task at once."""
        with self._lock:
            out = self._completed_store_tasks
            self._completed_store_tasks = {}
            return out

    # ------------------------------------------------------------------
    # Lookup-and-lock path
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Submit an asynchronous lookup-and-lock task.

        Returns a bitmap (via :meth:`query_lookup_and_lock_result`)
        where bit ``i`` is set when key ``i`` is present + pinned.
        Maru's pin contract is prefix-stop (the server stops at the
        first miss) — the bitmap reflects that.
        """
        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_lookup_tasks += 1

        assert self._lookup_executor is not None
        self._lookup_executor.submit(self._execute_lookup_task, task_id, keys)
        return task_id

    def _execute_lookup_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
    ) -> None:
        """Worker entry: ``batch_pin`` + record prefix-bitmap."""
        bitmap = Bitmap(len(keys))
        try:
            key_strs = [_object_key_to_string(k) for k in keys]
            pin_results = self._handler.batch_pin(key_strs)
            # Prefix-stop: first miss ends the contiguous hit run.
            for i, ok in enumerate(pin_results):
                if not ok:
                    break
                bitmap.set(i)
        except Exception:
            logger.exception(
                "MaruL2Adapter: lookup task %d failed (keys=%d)",
                task_id,
                len(keys),
            )

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
            self._inflight_lookup_tasks -= 1

        self._signal_eventfd(self._lookup_efd)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Pop the bitmap for ``task_id`` (single-consumer)."""
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release prior ``submit_lookup_and_lock_task`` locks.

        Synchronous — there is no per-task completion contract for
        unlock (the controller fires it and moves on).
        """
        if not keys:
            return
        key_strs = [_object_key_to_string(k) for k in keys]
        try:
            self._handler.batch_unpin(key_strs)
        except Exception:
            logger.exception("MaruL2Adapter: batch_unpin failed (keys=%d)", len(keys))

    # ------------------------------------------------------------------
    # Load path
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous CXL→DRAM load task.

        Args:
            keys: Keys whose payloads to fetch from MaruServer.
            objects: Pre-allocated L1 (DRAM) destination ``MemoryObj``\\ s.

        Returns:
            Task id usable with :meth:`query_load_result`.
        """
        if len(keys) != len(objects):
            raise ValueError(
                f"keys and objects length mismatch ({len(keys)} vs {len(objects)})"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_load_tasks += 1

        assert self._load_executor is not None
        self._load_executor.submit(self._execute_load_task, task_id, keys, objects)
        return task_id

    def _execute_load_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        """Worker entry: ``batch_retrieve`` + memcpy CXL→DRAM per key."""
        bitmap = Bitmap(len(keys))
        accessed: list[ObjectKey] = []
        try:
            key_strs = [_object_key_to_string(k) for k in keys]
            mem_infos = self._handler.batch_retrieve(key_strs)
            for i, (obj, info) in enumerate(zip(objects, mem_infos, strict=False)):
                if info is None:
                    continue
                nbytes = len(info.view)
                if nbytes <= 0:
                    continue
                src_addr = _memoryview_addr(info.view)
                ctypes.memmove(obj.data_ptr, src_addr, nbytes)
                bitmap.set(i)
                accessed.append(keys[i])
        except Exception:
            logger.exception(
                "MaruL2Adapter: load task %d failed (keys=%d)",
                task_id,
                len(keys),
            )

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
            self._inflight_load_tasks -= 1

        if accessed:
            self._notify_keys_accessed(accessed)
        self._signal_eventfd(self._load_efd)

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Pop the bitmap for ``task_id`` (single-consumer)."""
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Remove ``keys`` from MaruServer.

        ``MaruHandler.delete`` is per-key and may return ``False`` when
        the key is pinned (still being read) or missing. Both cases
        are logged but not re-raised — eviction is best-effort and the
        controller can retry later.
        """
        if not keys:
            return
        for key in keys:
            key_str = _object_key_to_string(key)
            try:
                self._handler.delete(key_str)
            except Exception:
                logger.exception("MaruL2Adapter: delete failed for key=%s", key_str)

    def close(self) -> None:
        """Shut down worker pools, signal final eventfd events, drop
        the MaruHandler connection. Best-effort: errors are logged but
        do not propagate (mirroring the base adapters)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        for executor_attr in (
            "_store_executor",
            "_lookup_executor",
            "_load_executor",
        ):
            executor = getattr(self, executor_attr)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                setattr(self, executor_attr, None)

        for efd_attr in ("_store_efd", "_lookup_efd", "_load_efd"):
            efd = getattr(self, efd_attr, None)
            if efd is not None:
                try:
                    efd.close()
                except OSError:
                    logger.debug("Skipping %s.close() — already closed", efd_attr)

        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:
                logger.exception("[MaruL2Adapter] MaruHandler.close() failed")
            self._handler = None


# ----------------------------------------------------------------------
# Factory registration
# ----------------------------------------------------------------------


def _create_maru_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Factory invoked by the L2 adapter registry.

    Args:
        config: Validated maru L2 config (the registry calls us with
            the base type; the concrete type is enforced at
            registration time by ``register_l2_adapter_type``).
        l1_memory_desc: L1 buffer descriptor passed by ``StorageManager``.
            Not used by this adapter — CXL ↔ DRAM transfer happens
            via ``MemoryObj.data_ptr`` on the inbound ``MemoryObj``,
            so no RDMA registration of a single L1 base pointer is
            required.
    """
    del l1_memory_desc
    return MaruL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_type("maru", MaruL2AdapterConfig)
register_l2_adapter_factory("maru", _create_maru_l2_adapter)
