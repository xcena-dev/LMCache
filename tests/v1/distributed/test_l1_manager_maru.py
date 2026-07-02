# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the maru sibling L1 manager (``MaruL1Manager``).

``MaruL1Manager`` is a standalone sibling of ``L1Manager`` selected at
``StorageManager`` when ``maru_config`` is set. It owns a
``MaruMemoryAllocator`` directly and delegates the RPC-driven operations
to an internal :class:`MaruL1Dispatcher`.

Coverage:

1. ``object_key_to_string`` — stable string form for MaruHandler RPCs.
2. ``MaruL1Manager.__init__`` — builds a ``MaruMemoryAllocator`` and
   wraps a :class:`MaruL1Dispatcher`.
3. STORE path: ``reserve_write`` (allocate) → ``finish_write``
   (``MaruHandler.batch_store``).
4. RETRIEVE path: ``reserve_read`` (``batch_pin`` + ``batch_retrieve`` +
   ``get_by_location`` + side channel) → ``unsafe_read`` (side channel
   lookup) → ``finish_read`` (``batch_unpin`` + side channel clear).
5. Race-condition rollback in ``reserve_read``.
6. ``delete`` / ``clear`` / ``finish_write_and_reserve_read``.
7. Parity methods: ``register_listener`` / ``touch_keys`` /
   ``is_key_evictable`` / ``memcheck`` / ``get_object_state`` /
   ``get_l1_memory_desc``.
8. ``report_status`` shape in maru mode.

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

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error

try:
    # First Party
    from lmcache.v1.distributed.maru_l1_dispatch import (
        MaruL1Dispatcher,
        _PendingRead,
        object_key_to_string,
    )
    from lmcache.v1.distributed.maru_l1_manager import MaruL1Manager
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
except ImportError:
    pytest.skip(
        "maru_l1_manager / maru_memory_allocator could not be imported",
        allow_module_level=True,
    )


# =========================================================================
# Fixtures
# =========================================================================


@dataclass
class _FakeMemInfo:
    """Minimal stand-in for ``MaruHandler.batch_retrieve`` return entries.

    Only the fields read by the dispatcher's ``reserve_read`` are present:
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
        instance_id="test-mp",
    )


@pytest.fixture
def fake_maru_allocator():
    """Replace ``MaruMemoryAllocator.__init__`` with a stub that
    installs ``MagicMock`` handler + adapter directly — equivalent
    to the post-``init_layout`` state. Reverts on teardown.
    """
    real_init = MaruMemoryAllocator.__init__

    def fake_init(self, config: MaruL1Config) -> None:
        real_init(self, config)
        # Post-``init_layout`` state: pool, handler, and layout
        # metadata are present so the allocator is considered
        # initialized.
        self._handler = mock.MagicMock(name="MaruHandler")
        self._cxl_adapter = mock.MagicMock(name="CxlMemoryAdapter")
        self._single_token_size = 4096  # dummy non-zero

    MaruMemoryAllocator.__init__ = fake_init
    try:
        yield
    finally:
        MaruMemoryAllocator.__init__ = real_init


@pytest.fixture
def maru_mgr(maru_cfg, fake_maru_allocator) -> MaruL1Manager:
    """Build a ``MaruL1Manager`` whose underlying allocator is a fake
    ``MaruMemoryAllocator``.
    """
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
    )
    return MaruL1Manager(cfg)


@pytest.fixture
def maru_handler(maru_mgr):
    """The ``MagicMock`` handler installed by ``fake_maru_allocator``.

    Shortcut so test methods can configure RPC return values directly
    without threading through the full dispatcher → allocator chain.
    """
    return maru_mgr._dispatcher._allocator._handler


@pytest.fixture
def maru_adapter(maru_mgr):
    """The ``MagicMock`` ``CxlMemoryAdapter`` installed by
    ``fake_maru_allocator``. Use it to configure
    ``create_store_handle`` / ``get_by_location`` side effects.
    """
    return maru_mgr._dispatcher._allocator._cxl_adapter


def _mk_key(idx: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, byteorder="big"),
        model_name="test-model",
        kv_rank=0xABCD,
        cache_salt=salt,
    )


def _seed_read(dispatcher, key, mem_obj=None, refcount=1):
    """Stage a read entry the way ``reserve_read`` would, with a refcount.

    The read side channel now holds :class:`_PendingRead` tuples rather
    than bare ``MemoryObj``s, so tests that poke it directly build one
    through this helper.
    """
    if mem_obj is None:
        mem_obj = mock.MagicMock()
    dispatcher._pending_read_memobjs[key] = _PendingRead(
        mem_obj=mem_obj, refcount=refcount
    )
    return mem_obj


# =========================================================================
# (1) object_key_to_string
# =========================================================================


class TestObjectKeyToString:
    def test_basic(self):
        k = _mk_key(idx=0x01020304)
        assert object_key_to_string(k) == "test-model@0000abcd@01020304"

    def test_with_salt(self):
        k = _mk_key(idx=0xFF, salt="user-1")
        assert object_key_to_string(k) == "test-model@0000abcd@000000ff@user-1"


# =========================================================================
# (2) __init__ / dispatcher wiring
# =========================================================================


class TestMaruBackendDetection:
    def test_dispatcher_wired(self, maru_mgr, maru_handler):
        assert isinstance(maru_mgr._dispatcher, MaruL1Dispatcher)
        # Dispatcher resolves ``handler`` through its allocator
        # reference — verify it points at the same MagicMock.
        assert maru_mgr._dispatcher.handler is maru_handler

    def test_pending_read_memobjs_initialized(self, maru_mgr):
        assert maru_mgr._dispatcher._pending_read_memobjs == {}


# =========================================================================
# (3) STORE path: reserve_write + finish_write
# =========================================================================


class TestMaruReserveWrite:
    def test_happy_path_allocates_via_memory_manager(self, maru_mgr):
        # Patch the memory_manager to short-circuit to a known result.
        keys = [_mk_key(i) for i in range(3)]
        fake_objs = [mock.MagicMock(spec=[]) for _ in keys]
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._dispatcher._memory_manager = maru_mgr._memory_manager
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

    def test_out_of_memory(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._dispatcher._memory_manager = maru_mgr._memory_manager
        maru_mgr._memory_manager.allocate.return_value = (L1Error.OUT_OF_MEMORY, [])

        ret = maru_mgr.reserve_write(
            keys, is_temporary=[False, False], layout_desc=mock.MagicMock(), mode="new"
        )
        for k in keys:
            err, returned = ret[k]
            assert err is L1Error.OUT_OF_MEMORY
            assert returned is None

    def test_populates_write_side_channel(self, maru_mgr):
        # reserve_write must stash the reserved objs so the keys-only
        # finish_write drain can recover them.
        keys = [_mk_key(i) for i in range(2)]
        fake_objs = [mock.MagicMock(spec=[]) for _ in keys]
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._dispatcher._memory_manager = maru_mgr._memory_manager
        maru_mgr._memory_manager.allocate.return_value = (L1Error.SUCCESS, fake_objs)

        maru_mgr.reserve_write(
            keys,
            is_temporary=[False] * len(keys),
            layout_desc=mock.MagicMock(),
            mode="new",
        )

        chan = maru_mgr._dispatcher._pending_write_memobjs
        for k, obj in zip(keys, fake_objs, strict=False):
            assert chan[k] is obj


class TestMaruFinishWrite:
    # finish_write is keys-only post-merge: the reserved MemoryObjs are
    # recovered from the write side channel that reserve_write populates.
    @staticmethod
    def _seed(maru_mgr, keys, memory_objs):
        """Populate the write side channel as reserve_write would."""
        maru_mgr._dispatcher._pending_write_memobjs = dict(
            zip(keys, memory_objs, strict=True)
        )

    def test_happy_path_calls_batch_store(self, maru_mgr, maru_handler, maru_adapter):
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock(name=f"mo-{i}") for i in range(2)]
        self._seed(maru_mgr, keys, memory_objs)
        # ``MaruMemoryAllocator.create_store_handle`` forwards to the
        # underlying ``CxlMemoryAdapter`` — we configure that mock to
        # observe and override the return values.
        handles = [mock.MagicMock(name=f"handle-{i}") for i in range(2)]
        maru_adapter.create_store_handle.side_effect = handles
        maru_handler.batch_store.return_value = [True, True]

        ret = maru_mgr.finish_write(keys)

        # Verify batch_store was called with key strings + handles.
        called_key_strs, called_handles = maru_handler.batch_store.call_args.args
        assert called_key_strs == [object_key_to_string(k) for k in keys]
        assert called_handles == handles
        for k in keys:
            assert ret[k] is L1Error.SUCCESS
        # The side channel is drained on success.
        assert maru_mgr._dispatcher._pending_write_memobjs == {}

    def test_dup_skip_returns_success(self, maru_mgr, maru_handler, maru_adapter):
        # ``batch_store`` returns True for both newly registered AND
        # dup-skipped keys; both are functional successes.
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock() for _ in keys]
        self._seed(maru_mgr, keys, memory_objs)
        maru_adapter.create_store_handle.side_effect = [mock.MagicMock() for _ in keys]
        # MaruHandler returns True even for dup-skipped keys.
        maru_handler.batch_store.return_value = [True, True]

        ret = maru_mgr.finish_write(keys)
        for k in keys:
            assert ret[k] is L1Error.SUCCESS

    def test_missing_from_side_channel_returns_error(self, maru_mgr, maru_handler):
        # Keys never reserved (absent from the side channel): error, no RPC.
        keys = [_mk_key(i) for i in range(2)]
        ret = maru_mgr.finish_write(keys)
        for k in keys:
            assert ret[k] is L1Error.KEY_IN_WRONG_STATE
        maru_handler.batch_store.assert_not_called()

    def test_partial_side_channel(self, maru_mgr, maru_handler, maru_adapter):
        # Only some keys were reserved: present keys are stored, missing keys
        # report an error, and the channel is fully drained either way.
        keys = [_mk_key(i) for i in range(2)]
        present_obj = mock.MagicMock(name="mo-present")
        self._seed(maru_mgr, keys[:1], [present_obj])
        maru_adapter.create_store_handle.side_effect = [mock.MagicMock()]
        maru_handler.batch_store.return_value = [True]

        ret = maru_mgr.finish_write(keys)

        called_key_strs, _ = maru_handler.batch_store.call_args.args
        assert called_key_strs == [object_key_to_string(keys[0])]
        assert ret[keys[0]] is L1Error.SUCCESS
        assert ret[keys[1]] is L1Error.KEY_IN_WRONG_STATE
        assert maru_mgr._dispatcher._pending_write_memobjs == {}

    def test_batch_store_exception_returns_error(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        keys = [_mk_key(i) for i in range(2)]
        memory_objs = [mock.MagicMock() for _ in keys]
        self._seed(maru_mgr, keys, memory_objs)
        maru_adapter.create_store_handle.side_effect = [mock.MagicMock() for _ in keys]
        maru_handler.batch_store.side_effect = RuntimeError("rpc fail")

        ret = maru_mgr.finish_write(keys)
        for k in keys:
            assert ret[k] is L1Error.KEY_IN_WRONG_STATE
        # Even on RPC failure the side channel must be drained (no leak).
        assert maru_mgr._dispatcher._pending_write_memobjs == {}


# =========================================================================
# (4) RETRIEVE path: reserve_read + unsafe_read + finish_read
# =========================================================================


class TestMaruReserveRead:
    def test_all_hit(self, maru_mgr, maru_handler, maru_adapter):
        keys = [_mk_key(i) for i in range(3)]
        maru_handler.batch_pin.return_value = [
            True,
            True,
            True,
        ]
        mem_infos = [
            _FakeMemInfo(region_id=i, page_index=i, view=b"x" * 32) for i in range(3)
        ]
        maru_handler.batch_retrieve.return_value = mem_infos
        fake_objs = [mock.MagicMock(name=f"obj-{i}") for i in range(3)]
        maru_adapter.get_by_location.side_effect = fake_objs

        ret = maru_mgr.reserve_read(keys)

        for k, obj in zip(keys, fake_objs, strict=False):
            err, returned = ret[k]
            assert err is L1Error.SUCCESS
            assert returned is obj
            entry = maru_mgr._dispatcher._pending_read_memobjs[k]
            assert entry.mem_obj is obj
            assert entry.refcount == 1

    def test_prefix_miss(self, maru_mgr, maru_handler, maru_adapter):
        """``batch_pin`` reports prefix-stop: only k0, k1 are pinned."""
        keys = [_mk_key(i) for i in range(3)]
        maru_handler.batch_pin.return_value = [
            True,
            True,
            False,
        ]
        mem_infos = [_FakeMemInfo(i, i, b"x" * 32) for i in range(2)]
        maru_handler.batch_retrieve.return_value = mem_infos
        fake_objs = [mock.MagicMock(), mock.MagicMock()]
        maru_adapter.get_by_location.side_effect = fake_objs

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]][0] is L1Error.SUCCESS
        assert ret[keys[2]] == (L1Error.KEY_NOT_EXIST, None)
        assert keys[2] not in maru_mgr._dispatcher._pending_read_memobjs

    def test_all_miss(self, maru_mgr, maru_handler):
        keys = [_mk_key(i) for i in range(2)]
        maru_handler.batch_pin.return_value = [
            False,
            False,
        ]

        ret = maru_mgr.reserve_read(keys)

        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)
        maru_handler.batch_retrieve.assert_not_called()
        assert maru_mgr._dispatcher._pending_read_memobjs == {}

    def test_race_batch_retrieve_returns_none_mid_batch(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        """k0 resolves; k1 races and returns None — k1 is unpinned."""
        keys = [_mk_key(i) for i in range(2)]
        maru_handler.batch_pin.return_value = [
            True,
            True,
        ]
        maru_handler.batch_retrieve.return_value = [
            _FakeMemInfo(0, 0, b"x" * 32),
            None,
        ]
        maru_adapter.get_by_location.return_value = mock.MagicMock(name="obj-0")

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]] == (L1Error.KEY_NOT_EXIST, None)
        # Only k1's key string should be rolled back via batch_unpin.
        maru_handler.batch_unpin.assert_called_once_with(
            [object_key_to_string(keys[1])]
        )

    def test_race_get_by_location_returns_none(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        keys = [_mk_key(0)]
        maru_handler.batch_pin.return_value = [True]
        maru_handler.batch_retrieve.return_value = [_FakeMemInfo(0, 0, b"x" * 32)]
        maru_adapter.get_by_location.return_value = None

        ret = maru_mgr.reserve_read(keys)

        assert ret[keys[0]] == (L1Error.KEY_NOT_EXIST, None)
        maru_handler.batch_unpin.assert_called_once_with(
            [object_key_to_string(keys[0])]
        )

    def test_batch_pin_exception_returns_miss_for_all(self, maru_mgr, maru_handler):
        keys = [_mk_key(i) for i in range(2)]
        maru_handler.batch_pin.side_effect = RuntimeError("rpc fail")

        ret = maru_mgr.reserve_read(keys)
        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)

    def test_batch_retrieve_exception_rolls_back_pins(self, maru_mgr, maru_handler):
        keys = [_mk_key(i) for i in range(2)]
        maru_handler.batch_pin.return_value = [
            True,
            True,
        ]
        maru_handler.batch_retrieve.side_effect = RuntimeError("rpc fail")

        ret = maru_mgr.reserve_read(keys)
        for k in keys:
            assert ret[k] == (L1Error.KEY_NOT_EXIST, None)
        # Both pins should have been rolled back.
        maru_handler.batch_unpin.assert_called_once_with(
            [object_key_to_string(k) for k in keys]
        )


class TestMaruUnsafeRead:
    def test_returns_staged_memobj(self, maru_mgr):
        keys = [_mk_key(i) for i in range(2)]
        fake_objs = [mock.MagicMock(name=f"obj-{i}") for i in range(2)]
        for k, obj in zip(keys, fake_objs, strict=False):
            _seed_read(maru_mgr._dispatcher, k, mem_obj=obj)

        ret = maru_mgr.unsafe_read(keys)

        for k, obj in zip(keys, fake_objs, strict=False):
            err, returned = ret[k]
            assert err is L1Error.SUCCESS
            assert returned is obj

    def test_missing_key_returns_not_exist(self, maru_mgr):
        keys = [_mk_key(0), _mk_key(1)]
        _seed_read(maru_mgr._dispatcher, keys[0])

        ret = maru_mgr.unsafe_read(keys)

        assert ret[keys[0]][0] is L1Error.SUCCESS
        assert ret[keys[1]] == (L1Error.KEY_NOT_EXIST, None)


class TestMaruFinishRead:
    def test_pops_side_channel_and_unpins(self, maru_mgr, maru_handler):
        keys = [_mk_key(i) for i in range(2)]
        for k in keys:
            _seed_read(maru_mgr._dispatcher, k)

        ret = maru_mgr.finish_read(keys)

        for k in keys:
            assert ret[k] is L1Error.SUCCESS
            assert k not in maru_mgr._dispatcher._pending_read_memobjs
        maru_handler.batch_unpin.assert_called_once_with(
            [object_key_to_string(k) for k in keys]
        )

    def test_non_pending_key_returns_not_exist_and_skips_unpin(
        self, maru_mgr, maru_handler
    ):
        unknown = _mk_key(99)
        ret = maru_mgr.finish_read([unknown])
        assert ret[unknown] is L1Error.KEY_NOT_EXIST
        maru_handler.batch_unpin.assert_not_called()

    def test_unpin_exception_does_not_propagate(self, maru_mgr, maru_handler):
        keys = [_mk_key(0)]
        _seed_read(maru_mgr._dispatcher, keys[0])
        maru_handler.batch_unpin.side_effect = RuntimeError("rpc fail")

        # Should not raise; side channel is still cleared.
        ret = maru_mgr.finish_read(keys)
        assert ret[keys[0]] is L1Error.SUCCESS
        assert maru_mgr._dispatcher._pending_read_memobjs == {}


# =========================================================================
# (5) delete / clear / finish_write_and_reserve_read
# =========================================================================


class TestMaruDelete:
    def test_success(self, maru_mgr, maru_handler):
        keys = [_mk_key(i) for i in range(2)]
        maru_handler.delete.return_value = True
        ret = maru_mgr.delete(keys)
        assert all(v is L1Error.SUCCESS for v in ret.values())
        assert maru_handler.delete.call_count == 2

    def test_handler_returns_false_reports_not_exist(self, maru_mgr, maru_handler):
        """``MaruHandler.delete`` returns False for either missing or
        pinned keys; the dispatcher maps both to ``KEY_NOT_EXIST``.
        """
        keys = [_mk_key(0)]
        maru_handler.delete.return_value = False
        ret = maru_mgr.delete(keys)
        assert ret[keys[0]] is L1Error.KEY_NOT_EXIST

    def test_exception_reports_wrong_state(self, maru_mgr, maru_handler):
        keys = [_mk_key(0)]
        maru_handler.delete.side_effect = RuntimeError("rpc fail")
        ret = maru_mgr.delete(keys)
        assert ret[keys[0]] is L1Error.KEY_IN_WRONG_STATE


class TestMaruClear:
    def test_clear_drops_side_channel_only(self, maru_mgr, maru_handler):
        _seed_read(maru_mgr._dispatcher, _mk_key(0))
        _seed_read(maru_mgr._dispatcher, _mk_key(1))

        maru_mgr.clear()

        assert maru_mgr._dispatcher._pending_read_memobjs == {}
        # Stored data is untouched: clear balances pins, never deletes.
        maru_handler.delete.assert_not_called()

    def test_force_clear_also_drops_only_side_channel(self, maru_mgr, maru_handler):
        _seed_read(maru_mgr._dispatcher, _mk_key(0))

        maru_mgr.clear(force=True)

        assert maru_mgr._dispatcher._pending_read_memobjs == {}
        maru_handler.delete.assert_not_called()

    def test_clear_unpins_once_per_remaining_refcount(self, maru_mgr, maru_handler):
        # A key staged with refcount=3 (three overlapping reads dropped
        # without finish_read) must be unpinned three times so MaruServer's
        # pin_count balances, not once per key.
        k0 = _mk_key(0)
        k1 = _mk_key(1)
        _seed_read(maru_mgr._dispatcher, k0, refcount=3)
        _seed_read(maru_mgr._dispatcher, k1, refcount=1)

        maru_mgr.clear(force=True)

        assert maru_mgr._dispatcher._pending_read_memobjs == {}
        maru_handler.batch_unpin.assert_called_once()
        (unpinned,) = maru_handler.batch_unpin.call_args.args
        assert unpinned.count(object_key_to_string(k0)) == 3
        assert unpinned.count(object_key_to_string(k1)) == 1
        assert len(unpinned) == 4


class TestMaruFinishWriteAndReserveRead:
    def test_resolves_from_side_channel(self, maru_mgr):
        k = _mk_key(0)
        fake_obj = _seed_read(maru_mgr._dispatcher, k)
        ret = maru_mgr.finish_write_and_reserve_read([k])
        err, returned = ret[k]
        assert err is L1Error.SUCCESS
        assert returned is fake_obj

    def test_missing_key_returns_not_exist(self, maru_mgr):
        ret = maru_mgr.finish_write_and_reserve_read([_mk_key(0)])
        assert ret[_mk_key(0)] == (L1Error.KEY_NOT_EXIST, None)


# =========================================================================
# (6) Parity methods
# =========================================================================


class TestMaruParityMethods:
    def test_register_listener_is_stored(self, maru_mgr):
        # PR1 parity: listeners are accepted and stored (firing is a later
        # PR, once the L2 controller stack is wired for maru).
        listener = mock.MagicMock()
        maru_mgr.register_listener(listener)
        assert listener in maru_mgr._registered_listeners

    def test_touch_keys_is_noop(self, maru_mgr):
        listener = mock.MagicMock()
        # Even if a listener is registered, maru-mode ``touch_keys`` should
        # not call it (MaruServer owns eviction, no LRU bookkeeping).
        maru_mgr._registered_listeners.append(listener)
        maru_mgr.touch_keys([_mk_key(0)])
        listener.on_l1_keys_accessed.assert_not_called()

    def test_is_key_evictable_always_true(self, maru_mgr):
        # No L1EvictionController is registered in maru mode, but the
        # method still returns True so any defensive caller sees a
        # consistent answer.
        assert maru_mgr.is_key_evictable(_mk_key(0)) is True
        # Also for a key that happens to be in the side channel.
        _seed_read(maru_mgr._dispatcher, _mk_key(1))
        assert maru_mgr.is_key_evictable(_mk_key(1)) is True

    def test_memcheck_returns_true(self, maru_mgr):
        assert maru_mgr.memcheck() is True

    def test_get_object_state_returns_none(self, maru_mgr):
        assert maru_mgr.get_object_state(_mk_key(0)) is None

    def test_get_l1_memory_desc_returns_none(self, maru_mgr):
        # Copy-type L2 contract: maru has no single contiguous buffer to
        # describe, so the descriptor is ``None``.
        assert maru_mgr.get_l1_memory_desc() is None


# =========================================================================
# (7) report_status / close
# =========================================================================


class TestMaruReportStatus:
    def test_shape(self, maru_mgr):
        _seed_read(maru_mgr._dispatcher, _mk_key(0))
        # Memory manager get_memory_usage is exercised elsewhere; here we
        # only verify the maru-mode dict shape.
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._dispatcher._memory_manager = maru_mgr._memory_manager
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
        _seed_read(maru_mgr._dispatcher, _mk_key(0))
        maru_mgr._memory_manager = mock.MagicMock()
        maru_mgr._dispatcher._memory_manager = maru_mgr._memory_manager

        maru_mgr.close()

        assert maru_mgr._dispatcher._pending_read_memobjs == {}
        maru_mgr._memory_manager.close.assert_called_once()


# =========================================================================
# (8) E7 regression: multi-reader refcount on the read side channel
# =========================================================================


class TestMaruMultiReaderRefcount:
    """Overlapping reads of the SAME key must each pin remotely and each
    unpin remotely, sharing one staged ``MemoryObj`` (E7 defect).

    Before the fix the read side channel was single-slot per key: two
    overlapping reserve_reads pinned twice remotely but the second staging
    overwrote the slot, so the first finish_read popped it and unpinned
    once while the second found nothing -> only 1 unpin for 2 pins ->
    remote pin_count stuck > 0 forever.
    """

    @staticmethod
    def _configure_single_key_hit(maru_handler, maru_adapter, objs):
        """Make each ``reserve_read`` of one key hit, handing out ``objs``
        (one per successive ``get_by_location`` call)."""
        maru_handler.batch_pin.return_value = [True]
        maru_handler.batch_retrieve.return_value = [_FakeMemInfo(0, 0, b"x" * 32)]
        maru_adapter.get_by_location.side_effect = list(objs)

    def test_double_reserve_pins_twice_shares_one_memobj(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        key = _mk_key(0)
        obj1 = mock.MagicMock(name="obj1")
        obj2 = mock.MagicMock(name="obj2")
        self._configure_single_key_hit(maru_handler, maru_adapter, [obj1, obj2])

        r1 = maru_mgr.reserve_read([key])
        r2 = maru_mgr.reserve_read([key])

        # Each reserve_read issued its own remote pin (N reserves == N pins).
        assert maru_handler.batch_pin.call_count == 2
        # Both reserves return the SAME staged MemoryObj (the first one);
        # the second materialised view (obj2) is discarded.
        assert r1[key] == (L1Error.SUCCESS, obj1)
        assert r2[key] == (L1Error.SUCCESS, obj1)
        entry = maru_mgr._dispatcher._pending_read_memobjs[key]
        assert entry.mem_obj is obj1
        assert entry.refcount == 2

    def test_two_finishes_two_unpins_channel_empty_only_after_second(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        key = _mk_key(0)
        obj1 = mock.MagicMock(name="obj1")
        obj2 = mock.MagicMock(name="obj2")
        self._configure_single_key_hit(maru_handler, maru_adapter, [obj1, obj2])
        maru_mgr.reserve_read([key])
        maru_mgr.reserve_read([key])

        # First finish: one unpin, still staged (refcount drops 2 -> 1).
        f1 = maru_mgr.finish_read([key])
        assert f1[key] is L1Error.SUCCESS
        assert key in maru_mgr._dispatcher._pending_read_memobjs
        assert maru_mgr._dispatcher._pending_read_memobjs[key].refcount == 1
        assert maru_handler.batch_unpin.call_count == 1

        # unsafe_read BETWEEN the two finishes still returns the object.
        ur = maru_mgr.unsafe_read([key])
        assert ur[key] == (L1Error.SUCCESS, obj1)

        # Second finish: second unpin, channel now empty.
        f2 = maru_mgr.finish_read([key])
        assert f2[key] is L1Error.SUCCESS
        assert key not in maru_mgr._dispatcher._pending_read_memobjs
        assert maru_handler.batch_unpin.call_count == 2

    def test_pin_unpin_balance_end_to_end(self, maru_mgr, maru_handler, maru_adapter):
        # The whole point: total remote pins == total remote unpins.
        key = _mk_key(0)
        self._configure_single_key_hit(
            maru_handler, maru_adapter, [mock.MagicMock(), mock.MagicMock()]
        )
        maru_mgr.reserve_read([key])
        maru_mgr.reserve_read([key])
        maru_mgr.finish_read([key])
        maru_mgr.finish_read([key])

        pins = sum(len(c.args[0]) for c in maru_handler.batch_pin.call_args_list)
        unpins = sum(len(c.args[0]) for c in maru_handler.batch_unpin.call_args_list)
        assert pins == unpins == 2
        assert maru_mgr._dispatcher._pending_read_memobjs == {}


class TestMaruReadThreadSafety:
    """Smoke test: concurrent reserve_read/finish_read loops on overlapping
    keys must leave the side channel empty and pins balanced by unpins.

    The manager lock serialises every side-channel touch, so the mock
    handler is never called from two threads at once and its call counts
    are reliable.
    """

    def test_concurrent_reserve_finish_balances(
        self, maru_mgr, maru_handler, maru_adapter
    ):
        # Standard
        import threading

        keys = [_mk_key(i) for i in range(3)]
        maru_handler.batch_pin.side_effect = lambda ks: [True] * len(ks)
        maru_handler.batch_retrieve.side_effect = lambda ks: [
            _FakeMemInfo(0, 0, b"x" * 32) for _ in ks
        ]
        maru_adapter.get_by_location.side_effect = lambda **kw: mock.MagicMock()

        iterations = 200
        num_threads = 2
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(iterations):
                    for k in keys:
                        maru_mgr.reserve_read([k])
                    for k in keys:
                        maru_mgr.finish_read([k])
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Every staged read was finished: side channel fully drained.
        assert maru_mgr._dispatcher._pending_read_memobjs == {}
        # Every remote pin was balanced by exactly one remote unpin.
        pins = sum(len(c.args[0]) for c in maru_handler.batch_pin.call_args_list)
        unpins = sum(len(c.args[0]) for c in maru_handler.batch_unpin.call_args_list)
        assert pins == unpins
        assert pins == num_threads * iterations * len(keys)


# =========================================================================
# Quick coverage for an unused-import suppressor — keep linters happy.
# =========================================================================


def test_optional_import_smoke():
    assert Optional[int] is not None  # noqa: B015
