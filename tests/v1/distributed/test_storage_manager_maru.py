# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the maru-backend wiring of StorageManager.

Coverage:

1. ``StorageManager.__init__`` in maru mode — controllers and L2
   adapters are not constructed; ``_is_maru`` is True.
2. ``register_kv_layout`` — forwards down through L1Manager and
   L1MemoryManager to ``MaruMemoryAllocator.init_layout``.
3. ``finish_write`` — forwards keys (keys-only) to ``L1Manager.finish_write``.
4. ``close()`` — succeeds with controllers absent.
5. ``report_status()`` — returns a maru-shaped dict.

The maru runtime (``maru``, ``maru_lmcache``) is NOT required: the
lazy ``MaruMemoryAllocator.__init__`` performs no RPC. Tests that
need a "pool ready" allocator monkey-patch ``init_layout`` so no
MaruServer connection is attempted.
"""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus

try:
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    pytest.skip(
        "storage_manager / maru_memory_allocator could not be imported",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _event_bus():
    """Initialize a minimal event bus for every test in this module
    (the storage manager publishes events at construction time).
    """
    init_event_bus(EventBusConfig(enabled=False))
    yield


@pytest.fixture
def maru_storage_config() -> StorageManagerConfig:
    """A ``StorageManagerConfig`` with the maru L1 backend selected."""
    maru_cfg = MaruL1Config(
        server_url="maru://localhost:5555",
        pool_size_bytes=1 << 30,
        instance_id="test-mp",
    )
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
    )
    l1_manager_config = L1ManagerConfig(memory_config=memory_config)
    return StorageManagerConfig(
        l1_manager_config=l1_manager_config,
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig([]),
    )


@pytest.fixture
def fake_maru_init_layout():
    """Replace ``MaruMemoryAllocator.init_layout`` with a stub that
    installs ``MagicMock`` handler + adapter — equivalent to the
    post-connect state. Reverts on teardown.
    """
    real_init_layout = MaruMemoryAllocator.init_layout

    def fake_init_layout(self, shapes, dtypes, fmt, chunk_size_in_tokens):
        if self._cxl_adapter is not None:
            # Layout-mismatch validation is independent of the
            # MaruServer connection, so defer to the real method when
            # a layout is already bound.
            real_init_layout(self, shapes, dtypes, fmt, chunk_size_in_tokens)
            return
        self._handler = mock.MagicMock(name="MaruHandler")
        self._cxl_adapter = mock.MagicMock(name="CxlMemoryAdapter")
        self._shapes = shapes
        self._dtypes = dtypes
        self._fmt = fmt
        self._chunk_size_in_tokens = chunk_size_in_tokens
        self._single_token_size = 4096

    MaruMemoryAllocator.init_layout = fake_init_layout
    try:
        yield
    finally:
        MaruMemoryAllocator.init_layout = real_init_layout


# =========================================================================
# (1) Maru-mode StorageManager construction
# =========================================================================


class TestStorageManagerMaruInit:
    def test_no_controllers_constructed(self, maru_storage_config):
        mgr = StorageManager(maru_storage_config)
        try:
            assert mgr._is_maru is True
            assert mgr._eviction_controller is None
            assert mgr._l2_eviction_controller is None
            assert mgr._store_controller is None
            assert mgr._prefetch_controller is None
            # ``_l2_adapters`` is a dict keyed by adapter_id since the dev
            # L2 add/delete refactor; maru keeps it empty.
            assert mgr._l2_adapters == {}
        finally:
            mgr.close()

    def test_quota_manager_present(self, maru_storage_config):
        # The HTTP layer expects a stable quota_manager reference.
        mgr = StorageManager(maru_storage_config)
        try:
            assert mgr.quota_manager is not None
        finally:
            mgr.close()


# =========================================================================
# (2) register_kv_layout chain
# =========================================================================


class TestRegisterKvLayoutChain:
    def test_forwards_to_allocator(self, maru_storage_config, fake_maru_init_layout):
        mgr = StorageManager(maru_storage_config)
        try:
            shapes = [torch.Size([2, 32, 256, 128])]
            dtypes = [torch.float16]
            mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256, 1)

            alloc = mgr._l1_manager._memory_manager._allocator
            assert alloc._shapes == shapes
            assert alloc._dtypes == dtypes
            assert alloc._fmt is MemoryFormat.KV_2LTD
            assert alloc._chunk_size_in_tokens == 256
            assert alloc._handler is not None
            assert alloc._cxl_adapter is not None
        finally:
            mgr.close()

    def test_layout_mismatch_raises(self, maru_storage_config, fake_maru_init_layout):
        mgr = StorageManager(maru_storage_config)
        try:
            shapes_a = [torch.Size([2, 32, 256, 128])]
            shapes_b = [torch.Size([2, 32, 128, 128])]  # different
            dtypes = [torch.float16]
            mgr.register_kv_layout(shapes_a, dtypes, MemoryFormat.KV_2LTD, 256, 1)
            with pytest.raises(ValueError, match="layout mismatch"):
                mgr.register_kv_layout(shapes_b, dtypes, MemoryFormat.KV_2LTD, 256, 1)
        finally:
            mgr.close()

    def test_same_layout_is_idempotent(
        self, maru_storage_config, fake_maru_init_layout
    ):
        mgr = StorageManager(maru_storage_config)
        try:
            shapes = [torch.Size([2, 32, 256, 128])]
            dtypes = [torch.float16]
            mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256, 1)
            # Second call with the same layout: no exception.
            mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256, 1)
        finally:
            mgr.close()

    def test_multi_object_group_rejected(
        self, maru_storage_config, fake_maru_init_layout
    ):
        # maru only forwards object group 0's layout, so it rejects models
        # with more than one object group.
        mgr = StorageManager(maru_storage_config)
        try:
            shapes = [torch.Size([2, 32, 256, 128])]
            dtypes = [torch.float16]
            with pytest.raises(ValueError, match="single object group"):
                mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256, 2)
        finally:
            mgr.close()


# =========================================================================
# (3) finish_write forwards keys (keys-only post-merge)
# =========================================================================


class TestFinishWriteForwarding:
    def test_forwards_keys_to_l1_manager(
        self, maru_storage_config, fake_maru_init_layout
    ):
        mgr = StorageManager(maru_storage_config)
        try:
            keys = [mock.MagicMock(name=f"k-{i}") for i in range(3)]
            with mock.patch.object(
                mgr._l1_manager, "finish_write", return_value={}
            ) as patched:
                mgr.finish_write(keys)
            # Keys-only: the maru store now recovers MemoryObjs from its own
            # write side channel, so no memory_objs are threaded here.
            patched.assert_called_once_with(keys)
        finally:
            mgr.close()


# =========================================================================
# (4) close() and report_status() without controllers
# =========================================================================


class TestCloseAndReport:
    def test_close_succeeds_without_controllers(self, maru_storage_config):
        mgr = StorageManager(maru_storage_config)
        # Should not raise even though all controllers are ``None``.
        mgr.close()

    def test_report_status_shape(self, maru_storage_config):
        mgr = StorageManager(maru_storage_config)
        try:
            status = mgr.report_status()
            assert status["backend"] == "maru"
            assert status["num_l2_adapters"] == 0
            assert status["l2_adapters"] == []
            assert "l1_manager" in status
            # Controller entries should not be present in maru mode.
            assert "store_controller" not in status
            assert "prefetch_controller" not in status
            assert "l1_eviction_controller" not in status
            assert "l2_eviction_controller" not in status
        finally:
            mgr.close()
