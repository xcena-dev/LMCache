# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``MaruL2Adapter``.

Coverage:

1. ``MaruL2AdapterConfig.from_dict`` — required-field validation,
   type checks, default fills.
2. Factory registration in the ``L2`` adapter registry.
3. ``submit_store_task`` — happy path (alloc + memmove DRAM→CXL +
   ``batch_store``), length mismatch, handler exception.
4. ``submit_lookup_and_lock_task`` — prefix-stop bitmap (all-hit,
   prefix, all-miss), handler exception.
5. ``submit_load_task`` — happy path memmove CXL→DRAM, partial-miss
   bitmap, handler exception.
6. ``submit_unlock`` / ``delete`` — sync dispatch, error swallowing,
   empty input.
7. ``close()`` — idempotent teardown without inflight work.
8. ``_object_key_to_string`` — encoding parity with the L1 dispatcher.

``MaruHandler`` is monkey-patched (``MaruL2Adapter._create_handler``
returns a ``MagicMock``) so no MaruServer is needed. Worker pools
are replaced with an inline ``_SyncExecutor`` so each ``submit_*``
returns deterministically before assertions.
"""

# Standard
from unittest import mock

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey

try:
    # First Party
    from lmcache.v1.distributed.l2_adapters.maru_l2_adapter import (
        MaruL2Adapter,
        MaruL2AdapterConfig,
        _memoryview_addr,
        _object_key_to_string,
    )
except ImportError:
    pytest.skip("MaruL2Adapter could not be imported", allow_module_level=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _SyncExecutor:
    """Inline replacement for ``ThreadPoolExecutor`` so worker
    callbacks run on the caller's thread — keeps assertions
    deterministic without ``shutdown(wait=True)`` plumbing.
    """

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return mock.MagicMock()

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        del wait, cancel_futures


class _FakeAllocHandle:
    """Stand-in for ``MaruHandler.AllocHandle`` — exposes ``.buf`` as
    a writable memoryview backed by a numpy uint8 array."""

    def __init__(self, size: int) -> None:
        self._arr = np.zeros(size, dtype=np.uint8)

    @property
    def buf(self) -> memoryview:
        return memoryview(self._arr)


def _mk_key(idx: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, byteorder="big"),
        model_name="test-model",
        kv_rank=0xABCD,
        cache_salt=salt,
    )


def _make_dram_memory_obj(
    backing: np.ndarray, size_override: int | None = None
) -> mock.MagicMock:
    """Wrap a contiguous numpy buffer as a fake ``MemoryObj``.

    The MagicMock exposes ``data_ptr`` (raw address) and
    ``get_size()`` so :meth:`MaruL2Adapter._execute_store_task`
    treats it like a real L1 DRAM allocation.
    """
    mo = mock.MagicMock(name="MemoryObj")
    mo.data_ptr = int(backing.ctypes.data)
    mo.get_size = mock.MagicMock(
        return_value=size_override if size_override is not None else backing.nbytes
    )
    return mo


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def base_cfg() -> MaruL2AdapterConfig:
    return MaruL2AdapterConfig(
        server_url="maru://localhost:5555",
        pool_size_gb=1.0,
        chunk_size_bytes=4 * 1024 * 1024,
        instance_id="test-l2",
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )


@pytest.fixture
def fake_handler() -> mock.MagicMock:
    h = mock.MagicMock(name="MaruHandler")
    h.instance_id = "test-l2"
    return h


@pytest.fixture
def adapter(base_cfg, fake_handler):
    """Build a ``MaruL2Adapter`` with the handler + executors stubbed."""
    with mock.patch.object(MaruL2Adapter, "_create_handler", return_value=fake_handler):
        a = MaruL2Adapter(base_cfg)
    # Swap async executors for the inline sync stub.
    a._store_executor = _SyncExecutor()
    a._lookup_executor = _SyncExecutor()
    a._load_executor = _SyncExecutor()
    try:
        yield a
    finally:
        # ``close()`` walks every executor; ``_SyncExecutor.shutdown``
        # is a no-op, so this is cheap.
        a.close()


# =====================================================================
# (1) Config
# =====================================================================


class TestConfig:
    def test_from_dict_minimal(self):
        cfg = MaruL2AdapterConfig.from_dict(
            {
                "server_url": "maru://localhost:5555",
                "pool_size_gb": 1,
                "chunk_size_bytes": 4194304,
            }
        )
        assert cfg.server_url == "maru://localhost:5555"
        assert cfg.pool_size_gb == 1.0
        assert cfg.chunk_size_bytes == 4194304
        assert cfg.instance_id is None
        assert cfg.num_store_workers == 1
        assert cfg.num_lookup_workers == 1
        assert cfg.num_load_workers >= 1
        assert cfg.eager_map is True

    def test_from_dict_full(self):
        cfg = MaruL2AdapterConfig.from_dict(
            {
                "server_url": "tcp://m:1",
                "pool_size_gb": 0.5,
                "chunk_size_bytes": 65536,
                "instance_id": "client-x",
                "num_store_workers": 2,
                "num_lookup_workers": 3,
                "num_load_workers": 4,
                "timeout_ms": 1000,
                "use_async_rpc": False,
                "max_inflight": 8,
                "eager_map": False,
            }
        )
        assert cfg.instance_id == "client-x"
        assert cfg.num_store_workers == 2
        assert cfg.num_lookup_workers == 3
        assert cfg.num_load_workers == 4
        assert cfg.timeout_ms == 1000
        assert cfg.use_async_rpc is False
        assert cfg.max_inflight == 8
        assert cfg.eager_map is False

    def test_from_dict_missing_server_url(self):
        with pytest.raises(ValueError, match="server_url"):
            MaruL2AdapterConfig.from_dict({"pool_size_gb": 1, "chunk_size_bytes": 4096})

    def test_from_dict_empty_server_url(self):
        with pytest.raises(ValueError, match="server_url"):
            MaruL2AdapterConfig.from_dict(
                {
                    "server_url": "  ",
                    "pool_size_gb": 1,
                    "chunk_size_bytes": 4096,
                }
            )

    def test_from_dict_pool_zero_rejected(self):
        with pytest.raises(ValueError, match="pool_size_gb"):
            MaruL2AdapterConfig.from_dict(
                {
                    "server_url": "maru://x",
                    "pool_size_gb": 0,
                    "chunk_size_bytes": 4096,
                }
            )

    def test_from_dict_chunk_size_zero_rejected(self):
        with pytest.raises(ValueError, match="chunk_size_bytes"):
            MaruL2AdapterConfig.from_dict(
                {"server_url": "maru://x", "pool_size_gb": 1, "chunk_size_bytes": 0}
            )

    def test_from_dict_instance_id_non_string_rejected(self):
        with pytest.raises(ValueError, match="instance_id"):
            MaruL2AdapterConfig.from_dict(
                {
                    "server_url": "maru://x",
                    "pool_size_gb": 1,
                    "chunk_size_bytes": 4096,
                    "instance_id": 123,
                }
            )

    def test_help_text_mentions_required_fields(self):
        txt = MaruL2AdapterConfig.help()
        assert "server_url" in txt and "required" in txt
        assert "pool_size_gb" in txt
        assert "chunk_size_bytes" in txt


# =====================================================================
# (2) Factory registration
# =====================================================================


class TestRegistration:
    def test_maru_registered_in_supported_types(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            _L2_ADAPTER_CONFIG_REGISTRY,
        )

        # Importing the module triggered the ``register_l2_adapter_type`` call
        # at module-bottom; the registry should now know "maru".
        assert "maru" in _L2_ADAPTER_CONFIG_REGISTRY

    def test_factory_returns_adapter(self, base_cfg, fake_handler):
        # First Party
        from lmcache.v1.distributed.l2_adapters.maru_l2_adapter import (
            _create_maru_l2_adapter,
        )

        with mock.patch.object(
            MaruL2Adapter, "_create_handler", return_value=fake_handler
        ):
            a = _create_maru_l2_adapter(base_cfg, l1_memory_desc=None)
        try:
            assert isinstance(a, MaruL2Adapter)
        finally:
            a.close()


# =====================================================================
# (3) Lifecycle / event fds
# =====================================================================


class TestLifecycle:
    def test_event_fds_are_distinct(self, adapter):
        fds = {
            adapter.get_store_event_fd(),
            adapter.get_lookup_and_lock_event_fd(),
            adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_close_idempotent(self, base_cfg, fake_handler):
        with mock.patch.object(
            MaruL2Adapter, "_create_handler", return_value=fake_handler
        ):
            a = MaruL2Adapter(base_cfg)
        a._store_executor = _SyncExecutor()
        a._lookup_executor = _SyncExecutor()
        a._load_executor = _SyncExecutor()
        a.close()
        a.close()  # second call must be a safe no-op

    def test_submit_after_close_raises(self, base_cfg, fake_handler):
        with mock.patch.object(
            MaruL2Adapter, "_create_handler", return_value=fake_handler
        ):
            a = MaruL2Adapter(base_cfg)
        a._store_executor = _SyncExecutor()
        a._lookup_executor = _SyncExecutor()
        a._load_executor = _SyncExecutor()
        a.close()
        with pytest.raises(RuntimeError, match="closed"):
            a.submit_store_task([], [])


# =====================================================================
# (4) Store path
# =====================================================================


class TestStore:
    def test_keys_objects_length_mismatch(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_store_task([_mk_key(0), _mk_key(1)], [mock.MagicMock()])

    def test_happy_path_alloc_memmove_store(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(2)]
        size = 1024
        src_arrs = [np.full(size, 0xAB + i, dtype=np.uint8) for i in range(2)]
        mem_objs = [_make_dram_memory_obj(a) for a in src_arrs]

        handles = [_FakeAllocHandle(size) for _ in keys]
        fake_handler.alloc.side_effect = handles
        fake_handler.batch_store.return_value = [True, True]

        task_id = adapter.submit_store_task(keys, mem_objs)

        # Verify alloc + batch_store were issued exactly once.
        assert fake_handler.alloc.call_count == 2
        fake_handler.batch_store.assert_called_once()
        passed_keys, passed_handles = fake_handler.batch_store.call_args.args
        assert passed_keys == [_object_key_to_string(k) for k in keys]
        assert passed_handles == handles

        # DRAM bytes must have made it into the CXL handle's buffer.
        for h, src in zip(handles, src_arrs, strict=True):
            assert bytes(h.buf[:size]) == bytes(src.tobytes())

        # Completion bookkeeping.
        results = adapter.pop_completed_store_tasks()
        assert results == {task_id: True}

    def test_batch_store_partial_failure_marks_overall_false(
        self, adapter, fake_handler
    ):
        keys = [_mk_key(0), _mk_key(1)]
        size = 256
        srcs = [np.zeros(size, dtype=np.uint8) for _ in keys]
        objs = [_make_dram_memory_obj(s) for s in srcs]

        fake_handler.alloc.side_effect = [
            _FakeAllocHandle(size),
            _FakeAllocHandle(size),
        ]
        # One success, one failure.
        fake_handler.batch_store.return_value = [True, False]

        task_id = adapter.submit_store_task(keys, objs)
        assert adapter.pop_completed_store_tasks() == {task_id: False}

    def test_alloc_exception_marks_failure(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        size = 256
        src = np.zeros(size, dtype=np.uint8)
        objs = [_make_dram_memory_obj(src)]

        fake_handler.alloc.side_effect = RuntimeError("oom")
        task_id = adapter.submit_store_task(keys, objs)

        # batch_store must not have been reached.
        fake_handler.batch_store.assert_not_called()
        assert adapter.pop_completed_store_tasks() == {task_id: False}

    def test_pop_drains_completed_dict(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        size = 16
        src = np.zeros(size, dtype=np.uint8)
        fake_handler.alloc.return_value = _FakeAllocHandle(size)
        fake_handler.batch_store.return_value = [True]

        task_id = adapter.submit_store_task(keys, [_make_dram_memory_obj(src)])
        assert adapter.pop_completed_store_tasks() == {task_id: True}
        # Second pop yields nothing — single-consumer contract.
        assert adapter.pop_completed_store_tasks() == {}


# =====================================================================
# (5) Lookup-and-lock path
# =====================================================================


class TestLookup:
    def test_all_hit(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(3)]
        fake_handler.batch_pin.return_value = [True, True, True]
        task_id = adapter.submit_lookup_and_lock_task(keys)
        bm = adapter.query_lookup_and_lock_result(task_id)
        assert bm is not None
        assert [bm.test(i) for i in range(3)] == [True, True, True]

    def test_prefix_stop(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(4)]
        fake_handler.batch_pin.return_value = [True, True, False, True]
        task_id = adapter.submit_lookup_and_lock_task(keys)
        bm = adapter.query_lookup_and_lock_result(task_id)
        # Once a miss hits, the bitmap freezes — even the trailing True
        # after the first False must not be set.
        assert [bm.test(i) for i in range(4)] == [True, True, False, False]

    def test_all_miss(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(2)]
        fake_handler.batch_pin.return_value = [False, False]
        task_id = adapter.submit_lookup_and_lock_task(keys)
        bm = adapter.query_lookup_and_lock_result(task_id)
        assert [bm.test(i) for i in range(2)] == [False, False]

    def test_handler_exception_empty_bitmap(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        fake_handler.batch_pin.side_effect = RuntimeError("rpc fail")
        task_id = adapter.submit_lookup_and_lock_task(keys)
        bm = adapter.query_lookup_and_lock_result(task_id)
        assert bm is not None
        assert bm.test(0) is False

    def test_query_is_single_consumer(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        fake_handler.batch_pin.return_value = [True]
        task_id = adapter.submit_lookup_and_lock_task(keys)
        assert adapter.query_lookup_and_lock_result(task_id) is not None
        # Second query returns None.
        assert adapter.query_lookup_and_lock_result(task_id) is None


# =====================================================================
# (6) Load path
# =====================================================================


class TestLoad:
    def test_keys_objects_length_mismatch(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_load_task([_mk_key(0)], [])

    def test_happy_path_memmove_cxl_to_dram(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        size = 1024
        cxl_arr = np.full(size, 0xCD, dtype=np.uint8)
        dram_arr = np.zeros(size, dtype=np.uint8)

        mi = mock.MagicMock(name="MemoryInfo")
        mi.view = memoryview(cxl_arr)
        fake_handler.batch_retrieve.return_value = [mi]

        task_id = adapter.submit_load_task(keys, [_make_dram_memory_obj(dram_arr)])
        bm = adapter.query_load_result(task_id)
        assert bm is not None and bm.test(0) is True
        # CXL bytes must have landed in DRAM.
        assert bytes(dram_arr.tobytes()) == bytes(cxl_arr.tobytes())

    def test_partial_miss_bitmap(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(3)]
        size = 32
        cxl_arrs = [np.full(size, 0x10 + i, dtype=np.uint8) for i in range(3)]
        # Second slot is a miss — None from MaruServer.
        mem_infos = []
        for i, arr in enumerate(cxl_arrs):
            if i == 1:
                mem_infos.append(None)
            else:
                mi = mock.MagicMock()
                mi.view = memoryview(arr)
                mem_infos.append(mi)
        fake_handler.batch_retrieve.return_value = mem_infos

        dram_arrs = [np.zeros(size, dtype=np.uint8) for _ in keys]
        mem_objs = [_make_dram_memory_obj(a) for a in dram_arrs]

        task_id = adapter.submit_load_task(keys, mem_objs)
        bm = adapter.query_load_result(task_id)
        assert [bm.test(i) for i in range(3)] == [True, False, True]
        # Hit slots got the data; miss slot stayed zero.
        assert bytes(dram_arrs[0].tobytes()) == bytes(cxl_arrs[0].tobytes())
        assert int(dram_arrs[1].sum()) == 0
        assert bytes(dram_arrs[2].tobytes()) == bytes(cxl_arrs[2].tobytes())

    def test_handler_exception_empty_bitmap(self, adapter, fake_handler):
        keys = [_mk_key(0)]
        fake_handler.batch_retrieve.side_effect = RuntimeError("rpc fail")
        size = 16
        dram = np.zeros(size, dtype=np.uint8)
        task_id = adapter.submit_load_task(keys, [_make_dram_memory_obj(dram)])
        bm = adapter.query_load_result(task_id)
        assert bm is not None and bm.test(0) is False
        # DRAM untouched.
        assert int(dram.sum()) == 0


# =====================================================================
# (7) Unlock / delete
# =====================================================================


class TestUnlockDelete:
    def test_unlock_invokes_batch_unpin_with_encoded_keys(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(2)]
        adapter.submit_unlock(keys)
        called_keys = fake_handler.batch_unpin.call_args.args[0]
        assert called_keys == [_object_key_to_string(k) for k in keys]

    def test_unlock_empty_skips_rpc(self, adapter, fake_handler):
        adapter.submit_unlock([])
        fake_handler.batch_unpin.assert_not_called()

    def test_unlock_swallows_exception(self, adapter, fake_handler):
        fake_handler.batch_unpin.side_effect = RuntimeError("boom")
        adapter.submit_unlock([_mk_key(0)])  # must not raise

    def test_delete_calls_per_key(self, adapter, fake_handler):
        keys = [_mk_key(i) for i in range(3)]
        adapter.delete(keys)
        assert fake_handler.delete.call_count == 3
        called_keys = [call.args[0] for call in fake_handler.delete.call_args_list]
        assert called_keys == [_object_key_to_string(k) for k in keys]

    def test_delete_empty_skips_rpc(self, adapter, fake_handler):
        adapter.delete([])
        fake_handler.delete.assert_not_called()

    def test_delete_swallows_exception(self, adapter, fake_handler):
        fake_handler.delete.side_effect = RuntimeError("boom")
        adapter.delete([_mk_key(0), _mk_key(1)])  # must not raise
        # Both keys attempted regardless of the first failure.
        assert fake_handler.delete.call_count == 2


# =====================================================================
# (8) Key encoding
# =====================================================================


class TestKeyEncoding:
    def test_basic(self):
        k = ObjectKey(
            chunk_hash=(0x01020304).to_bytes(4, "big"),
            model_name="m",
            kv_rank=0xAB,
            cache_salt="",
        )
        assert _object_key_to_string(k) == "m@000000ab@01020304"

    def test_with_salt(self):
        k = ObjectKey(
            chunk_hash=(0xFF).to_bytes(4, "big"),
            model_name="m",
            kv_rank=0xAB,
            cache_salt="u1",
        )
        assert _object_key_to_string(k) == "m@000000ab@000000ff@u1"


# =====================================================================
# (9) _memoryview_addr
# =====================================================================


class TestMemoryviewAddr:
    def test_returns_positive_int(self):
        buf = bytearray(b"\x00" * 16)
        addr = _memoryview_addr(memoryview(buf))
        assert isinstance(addr, int) and addr > 0

    def test_stable_across_calls(self):
        # A fixed-size bytearray's pointer must not move between calls.
        buf = bytearray(b"\x00" * 16)
        mv = memoryview(buf)
        assert _memoryview_addr(mv) == _memoryview_addr(mv)
