# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the maru-backend branches of L1Manager.

These tests cover the Phase 1.C changes in
``docs/source/mp/maru/integration.md``:

1. ``_object_key_to_string`` — stable string form for MaruHandler RPCs.
2. ``L1Manager.__init__`` — auto-detects ``MaruMemoryAllocator`` and
   sets ``_maru_handler`` / ``_maru_allocator`` / side channel.
3. ``_is_maru_backend`` — dispatch flag.
4. STORE path: ``reserve_write`` (allocate) → ``finish_write``
   (``MaruHandler.batch_store``).
5. RETRIEVE path: ``reserve_read`` (``batch_pin`` + ``batch_retrieve`` +
   ``get_by_location`` + side channel) → ``unsafe_read`` (side channel
   lookup) → ``finish_read`` (``batch_unpin`` + side channel clear).
6. Race-condition rollback in ``reserve_read``.
7. ``delete`` / ``clear`` / ``finish_write_and_reserve_read``.
8. No-op methods: ``register_listener`` / ``touch_keys`` /
   ``is_key_evictable`` / ``memcheck`` / ``get_object_state``.
9. ``report_status`` shape in maru mode.

The maru runtime (``maru``, ``maru_lmcache``) is NOT required: the
``MaruMemoryAllocator`` constructor is monkey-patched to install
``MagicMock`` handler + adapter instead of opening a real connection.
"""

# Standard
from dataclasses import dataclass
from typing import Optional
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.memory_management import MemoryFormat

try:
    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager, _object_key_to_string
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
except ImportError:
    pytest.skip(
        "l1_manager / maru_memory_allocator could not be imported",
        allow_module_level=True,
    )


# =========================================================================
# Fixtures
# =========================================================================


@dataclass
class _FakeMemInfo:
    """Minimal stand-in for ``MaruHandler.batch_retrieve`` return entries.

    Only the fields read by ``L1Manager._maru_reserve_read`` are present:
    ``region_id``, ``page_index``, and a ``view`` object whose ``len()``
    gives the chunk size in bytes.
    """

    region_id: int
    page_index: int
    view: bytes


@pytest.fixture
def maru_cfg() -> MaruL1Config:
    return MaruL1Config(
        server_url="maru://localhost:5555",
        pool_size_bytes=60 * 1024**3,
        full_chunk_size_bytes=256 * 4096,
        chunk_size_in_tokens=256,
        shapes=[torch.Size([2, 32, 256, 128])],
        dtypes=[torch.float16],
        fmt=MemoryFormat.KV_2LTD,
        instance_id="test-mp",
    )


@pytest.fixture
def fake_maru_allocator():
    """Replace ``MaruMemoryAllocator.__init__`` with a stub that
    installs ``MagicMock`` handler + adapter. Reverts on teardown.
    """
    real_init = MaruMemoryAllocator.__init__

    def fake_init(self, config: MaruL1Config) -> None:
        self._config = config
        self._single_token_size = (
            config.full_chunk_size_bytes // config.chunk_size_in_tokens
        )
        self._handler = mock.MagicMock(name="MaruHandler")
        self._cxl_adapter = mock.MagicMock(name="CxlMemoryAdapter")

    MaruMemoryAllocator.__init__ = fake_init
    try:
        yield
    finally:
        MaruMemoryAllocator.__init__ = real_init


@pytest.fixture
def maru_mgr(maru_cfg, fake_maru_allocator) -> L1Manager:
    """Build an ``L1Manager`` whose underlying allocator is a fake
    ``MaruMemoryAllocator``.
    """
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
    )
    return L1Manager(cfg)


def _mk_key(idx: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, byteorder="big"),
        model_name="test-model",
        kv_rank=0xABCD,
        cache_salt=salt,
    )


# =========================================================================
# (1) _object_key_to_string
# =========================================================================


class TestObjectKeyToString:
    def test_basic(self):
        k = _mk_key(idx=0x01020304)
        assert _object_key_to_string(k) == "test-model@0000abcd@01020304"

    def test_with_salt(self):
        k = _mk_key(idx=0xFF, salt="user-1")
        assert _object_key_to_string(k) == "test-model@0000abcd@000000ff@user-1"


# =========================================================================
# (2) __init__ / _is_maru_backend
# =========================================================================


class TestMaruBackendDetection:
    def test_maru_handler_wired(self, maru_mgr):
        assert maru_mgr._is_maru_backend() is True
        assert maru_mgr._maru_handler is not None
        assert maru_mgr._maru_allocator is not None
        # ``handler`` is exposed by the allocator's mock; verify we
        # grabbed it during __init__.
        assert maru_mgr._maru_handler is maru_mgr._maru_allocator._handler

    def test_pending_read_memobjs_initialized(self, maru_mgr):
        assert maru_mgr._pending_read_memobjs == {}


# =========================================================================
# (3) STORE path: reserve_write + finish_write
# =========================================================================


class TestMaruReserveWrite:
    def test_happy_path_allocates_via_memory_manager(self, maru_mgr):
        # Patch the memory_manager to short-circuit to a known result.
        keys = [_mk_key(i) for i in range(3)]
        fake_objs = [mock.MagicMock(spec=[]) for _ in keys]
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._memory_manager.allocate.return_value = (L1Error.SUCCESS, fake_objs)

        ret = maru_mgr.reserve_write(
            keys,
            is_temporary=[False] * len(keys),
            layout_desc=mock.MagicMock(),
            mode="new",
        )

        assert maru_mgr._memory_manager.allocate.called
        for k, obj in zip(keys, fake_objs, strict=False):
            err, returned = ret[k]
            assert err is L1Error.SUCCESS
            assert returned is obj
        # No in-process dict entries should be created in maru mode.
        assert maru_mgr._objects == {}

    def test_out_of_memory(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._memory_manager.allocate.return_value = (L1Error.OUT_OF_MEMORY, [])

        ret = maru_mgr.reserve_write(
            keys, is_temporary=[False, False], layout_desc=mock.MagicMock(), mode="new"
        )
        for k in keys:
            err, returned = ret[k]
            assert err is L1Error.OUT_OF_MEMORY
            assert returned is None


class TestMaruFinishWrite:
    def test_happy_path_calls_batch_store(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock(name=f"mo-{i}") for i in range(2)]
        # ``MaruMemoryAllocator.create_store_handle`` forwards to the
        # underlying ``CxlMemoryAdapter`` — we configure that mock to
        # observe and override the return values.
        handles = [mock.MagicMock(name=f"handle-{i}") for i in range(2)]
        maru_mgr._maru_allocator._cxl_adapter.create_store_handle.side_effect = handles
        maru_mgr._maru_handler.batch_store.return_value = [True, True]

        ret = maru_mgr.finish_write(keys, memory_objs=memory_objs)

        # Verify batch_store was called with key strings + handles.
        called_args = maru_mgr._maru_handler.batch_store.call_args
        called_key_strs, called_handles = called_args.args
        assert called_key_strs == [_object_key_to_string(k) for k in keys]
        assert called_handles == handles
        for k in keys:
            assert ret[k] is L1Error.SUCCESS

    def test_dup_skip_returns_success(self, maru_mgr):
        # ``batch_store`` returns True for both newly registered AND
        # dup-skipped keys; both are functional successes.
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock() for _ in keys]
        maru_mgr._maru_allocator._cxl_adapter.create_store_handle.side_effect = [
            mock.MagicMock() for _ in keys
        ]
        # MaruHandler returns True even for dup-skipped keys.
        maru_mgr._maru_handler.batch_store.return_value = [True, True]

        ret = maru_mgr.finish_write(keys, memory_objs=memory_objs)
        for k in keys:
            assert ret[k] is L1Error.SUCCESS

    def test_missing_memory_objs_returns_error(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        ret = maru_mgr.finish_write(keys, memory_objs=None)
        for k in keys:
            assert ret[k] is L1Error.KEY_IN_WRONG_STATE
        maru_mgr._maru_handler.batch_store.assert_not_called()

    def test_length_mismatch_returns_error(self, maru_mgr):
        keys = [_mk_key(i) for i in range(3)]
        ret = maru_mgr.finish_write(keys, memory_objs=[mock.MagicMock()])
        for k in keys:
            assert ret[k] is L1Error.KEY_IN_WRONG_STATE
        maru_mgr._maru_handler.batch_store.assert_not_called()

    def test_batch_store_exception_returns_error(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock() for _ in keys]
        maru_mgr._maru_allocator._cxl_adapter.create_store_handle.side_effect = [
            mock.MagicMock() for _ in keys
        ]
        maru_mgr._maru_handler.batch_store.side_effect = RuntimeError("rpc fail")
        ret = maru_mgr.finish_write(keys, memory_objs=memory_objs)
        for k in keys:
            assert ret[k] is L1Error.KEY_IN_WRONG_STATE


# =========================================================================
# (4) RETRIEVE path: reserve_read + unsafe_read + finish_read
# =========================================================================


class TestMaruReserveRead:
    def test_all_hit(self, maru_mgr):
        keys = [_mk_key(i) for i in range(3)]
        maru_mgr._maru_handler.batch_pin.return_value = [True, True, True]
        mem_infos = [
            _FakeMemInfo(region_id=i, page_index=i, view=b"x" * 32) for i in range(3)
        ]
        maru_mgr._maru_handler.batch_retrieve.return_value = mem_infos
        fake_objs = [mock.MagicMock(name=f"obj-{i}") for i in range(3)]
        maru_mgr._maru_allocator._cxl_adapter.get_by_location.side_effect = fake_objs

        ret = maru_mgr.reserve_read(keys)

        for k, obj in zip(keys, fake_objs, strict=False):
            err, returned = ret[k]
            assert err is L1Error.SUCCESS
            assert returned is obj
            assert maru_mgr._pending_read_memobjs[k] is obj

    def test_prefix_miss(self, maru_mgr):
        """``batch_pin`` reports prefix-stop: only k0, k1 are pinned."""
        keys = [_mk_key(i) for i in range(3)]
        maru_mgr._maru_handler.batch_pin.return_value = [True, True, False]
        mem_infos = [_FakeMemInfo(i, i, b"x" * 32) for i in range(2)]
        maru_mgr._maru_handler.batch_retrieve.return_value = mem_infos
        fake_objs = [mock.MagicMock(), mock.MagicMock()]
        maru_mgr._maru_allocator._cxl_adapter.get_by_location.side_effect = fake_objs

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]][0] is L1Error.SUCCESS
        assert ret[keys[2]] == (L1Error.KEY_NOT_EXIST, None)
        assert keys[2] not in maru_mgr._pending_read_memobjs

    def test_all_miss(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._maru_handler.batch_pin.return_value = [False, False]

        ret = maru_mgr.reserve_read(keys)

        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)
        maru_mgr._maru_handler.batch_retrieve.assert_not_called()
        assert maru_mgr._pending_read_memobjs == {}

    def test_race_batch_retrieve_returns_none_mid_batch(self, maru_mgr):
        """k0 resolves; k1 races and returns None — k1 is unpinned."""
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._maru_handler.batch_pin.return_value = [True, True]
        maru_mgr._maru_handler.batch_retrieve.return_value = [
            _FakeMemInfo(0, 0, b"x" * 32),
            None,
        ]
        maru_mgr._maru_allocator._cxl_adapter.get_by_location.return_value = (
            mock.MagicMock(name="obj-0")
        )

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]] == (L1Error.KEY_NOT_EXIST, None)
        # Only k1's key string should be rolled back via batch_unpin.
        maru_mgr._maru_handler.batch_unpin.assert_called_once_with(
            [_object_key_to_string(keys[1])]
        )

    def test_race_get_by_location_returns_none(self, maru_mgr):
        keys = [_mk_key(0)]
        maru_mgr._maru_handler.batch_pin.return_value = [True]
        maru_mgr._maru_handler.batch_retrieve.return_value = [
            _FakeMemInfo(0, 0, b"x" * 32)
        ]
        maru_mgr._maru_allocator._cxl_adapter.get_by_location.return_value = None

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]] == (L1Error.KEY_NOT_EXIST, None)
        maru_mgr._maru_handler.batch_unpin.assert_called_once_with(
            [_object_key_to_string(keys[0])]
        )

    def test_batch_pin_exception_returns_miss_for_all(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._maru_handler.batch_pin.side_effect = RuntimeError("rpc fail")

        ret = maru_mgr.reserve_read(keys)
        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)

    def test_batch_retrieve_exception_rolls_back_pins(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._maru_handler.batch_pin.return_value = [True, True]
        maru_mgr._maru_handler.batch_retrieve.side_effect = RuntimeError("rpc fail")

        ret = maru_mgr.reserve_read(keys)
        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)
        # Both pins should have been rolled back.
        maru_mgr._maru_handler.batch_unpin.assert_called_once_with(
            [_object_key_to_string(k) for k in keys]
        )


class TestMaruUnsafeRead:
    def test_returns_staged_memobj(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        fake_objs = [mock.MagicMock(name=f"obj-{i}") for i in range(2)]
        for k, obj in zip(keys, fake_objs, strict=False):
            maru_mgr._pending_read_memobjs[k] = obj

        ret = maru_mgr.unsafe_read(keys)

        for k, obj in zip(keys, fake_objs, strict=False):
            err, returned = ret[k]
            assert err is L1Error.SUCCESS
            assert returned is obj

    def test_missing_key_returns_not_exist(self, maru_mgr):
        keys = [_mk_key(0), _mk_key(1)]
        maru_mgr._pending_read_memobjs[keys[0]] = mock.MagicMock()

        ret = maru_mgr.unsafe_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]] == (L1Error.KEY_NOT_EXIST, None)


class TestMaruFinishRead:
    def test_pops_side_channel_and_unpins(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        for k in keys:
            maru_mgr._pending_read_memobjs[k] = mock.MagicMock()

        ret = maru_mgr.finish_read(keys)

        for k in keys:
            assert ret[k] is L1Error.SUCCESS
            assert k not in maru_mgr._pending_read_memobjs
        maru_mgr._maru_handler.batch_unpin.assert_called_once_with(
            [_object_key_to_string(k) for k in keys]
        )

    def test_non_pending_key_returns_not_exist_and_skips_unpin(self, maru_mgr):
        unknown = _mk_key(99)
        ret = maru_mgr.finish_read([unknown])
        assert ret[unknown] is L1Error.KEY_NOT_EXIST
        maru_mgr._maru_handler.batch_unpin.assert_not_called()

    def test_unpin_exception_does_not_propagate(self, maru_mgr):
        keys = [_mk_key(0)]
        maru_mgr._pending_read_memobjs[keys[0]] = mock.MagicMock()
        maru_mgr._maru_handler.batch_unpin.side_effect = RuntimeError("rpc fail")

        # Should not raise; side channel is still cleared.
        ret = maru_mgr.finish_read(keys)
        assert ret[keys[0]] is L1Error.SUCCESS
        assert maru_mgr._pending_read_memobjs == {}


# =========================================================================
# (5) delete / clear / finish_write_and_reserve_read
# =========================================================================


class TestMaruDelete:
    def test_success(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._maru_handler.delete.return_value = True
        ret = maru_mgr.delete(keys)
        assert all(v is L1Error.SUCCESS for v in ret.values())
        assert maru_mgr._maru_handler.delete.call_count == 2

    def test_handler_returns_false_reports_not_exist(self, maru_mgr):
        """``MaruHandler.delete`` returns False for either missing or
        pinned keys; ``L1Manager`` maps both to ``KEY_NOT_EXIST``.
        """
        keys = [_mk_key(0)]
        maru_mgr._maru_handler.delete.return_value = False
        ret = maru_mgr.delete(keys)
        assert ret[keys[0]] is L1Error.KEY_NOT_EXIST

    def test_exception_reports_wrong_state(self, maru_mgr):
        keys = [_mk_key(0)]
        maru_mgr._maru_handler.delete.side_effect = RuntimeError("rpc fail")
        ret = maru_mgr.delete(keys)
        assert ret[keys[0]] is L1Error.KEY_IN_WRONG_STATE


class TestMaruClear:
    def test_clear_drops_side_channel_only(self, maru_mgr):
        maru_mgr._pending_read_memobjs[_mk_key(0)] = mock.MagicMock()
        maru_mgr._pending_read_memobjs[_mk_key(1)] = mock.MagicMock()

        maru_mgr.clear()

        assert maru_mgr._pending_read_memobjs == {}
        # Server-side state is untouched.
        maru_mgr._maru_handler.delete.assert_not_called()

    def test_force_clear_also_drops_only_side_channel(self, maru_mgr):
        maru_mgr._pending_read_memobjs[_mk_key(0)] = mock.MagicMock()

        maru_mgr.clear(force=True)

        assert maru_mgr._pending_read_memobjs == {}
        maru_mgr._maru_handler.delete.assert_not_called()


class TestMaruFinishWriteAndReserveRead:
    def test_resolves_from_side_channel(self, maru_mgr):
        k = _mk_key(0)
        fake_obj = mock.MagicMock()
        maru_mgr._pending_read_memobjs[k] = fake_obj
        ret = maru_mgr.finish_write_and_reserve_read([k])
        err, returned = ret[k]
        assert err is L1Error.SUCCESS
        assert returned is fake_obj

    def test_missing_key_returns_not_exist(self, maru_mgr):
        ret = maru_mgr.finish_write_and_reserve_read([_mk_key(0)])
        assert ret[_mk_key(0)] == (L1Error.KEY_NOT_EXIST, None)


# =========================================================================
# (6) No-op methods
# =========================================================================


class TestMaruNoOps:
    def test_register_listener_is_dropped(self, maru_mgr):
        listener = mock.MagicMock()
        maru_mgr.register_listener(listener)
        assert listener not in maru_mgr._registered_listeners

    def test_touch_keys_is_noop(self, maru_mgr):
        listener = mock.MagicMock()
        # Even if a listener somehow ended up registered, maru-mode
        # ``touch_keys`` should not call it.
        maru_mgr._registered_listeners.append(listener)
        maru_mgr.touch_keys([_mk_key(0)])
        listener.on_l1_keys_accessed.assert_not_called()

    def test_is_key_evictable_always_true(self, maru_mgr):
        # No L1EvictionController is registered in maru mode, but the
        # method still returns True so any defensive caller sees a
        # consistent answer.
        assert maru_mgr.is_key_evictable(_mk_key(0)) is True
        # Also for a key that happens to be in the side channel.
        maru_mgr._pending_read_memobjs[_mk_key(1)] = mock.MagicMock()
        assert maru_mgr.is_key_evictable(_mk_key(1)) is True

    def test_memcheck_returns_true(self, maru_mgr):
        assert maru_mgr.memcheck() is True

    def test_get_object_state_returns_none(self, maru_mgr):
        assert maru_mgr.get_object_state(_mk_key(0)) is None


# =========================================================================
# (7) report_status / close
# =========================================================================


class TestMaruReportStatus:
    def test_shape(self, maru_mgr):
        maru_mgr._pending_read_memobjs[_mk_key(0)] = mock.MagicMock()
        # Memory manager get_memory_usage is already covered by the
        # Phase 1.B tests; here we just verify the maru-mode dict shape.
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._memory_manager.get_memory_usage.return_value = (10, 100)

        status = maru_mgr.report_status()

        assert status["backend"] == "maru"
        assert status["is_healthy"] is True
        assert status["total_object_count"] == 0
        assert status["pending_read_memobjs"] == 1
        assert status["memory_used_bytes"] == 10
        assert status["memory_total_bytes"] == 100
        assert status["memory_usage_ratio"] == 0.1


class TestMaruClose:
    def test_close_clears_side_channel(self, maru_mgr):
        maru_mgr._pending_read_memobjs[_mk_key(0)] = mock.MagicMock()
        maru_mgr._memory_manager = mock.MagicMock()

        maru_mgr.close()

        assert maru_mgr._pending_read_memobjs == {}
        maru_mgr._memory_manager.close.assert_called_once()


# =========================================================================
# Quick coverage for an unused-import suppressor — keep linters happy.
# =========================================================================


def test_optional_import_smoke():
    assert Optional[int] is not None  # noqa: B015
