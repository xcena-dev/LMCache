# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the maru-backend wiring of L1MemoryManager.

Coverage:

1. ``L1MemoryManagerConfig.maru_config`` — when set, the DRAM-only
   ``init_size_in_bytes`` clamp is skipped.
2. ``create_memory_allocator()`` — routes to ``MaruMemoryAllocator``
   when ``maru_config`` is set, otherwise to the existing DRAM
   allocators.
3. ``_is_maru_allocator()`` helper.
4. ``L1MemoryManager.get_memory_usage()`` — best-effort forwarding to
   ``MaruHandler.get_stats``; short-circuits to ``(0, 0)`` before
   ``init_layout`` is called.
5. ``L1MemoryManager.get_l1_memory_desc()`` — raises
   ``NotImplementedError`` for maru (no contiguous DRAM buffer).
6. ``L1MemoryManager.register_kv_layout()`` — forwards to
   ``MaruMemoryAllocator.init_layout`` for the maru backend and is a
   no-op for default backends.

The maru runtime (``maru``, ``maru_lmcache``) is NOT required: the
lazy ``MaruMemoryAllocator.__init__`` performs no RPC. Tests that
need an "initialized" allocator install ``MagicMock`` handler +
adapter directly on the instance.
"""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat

try:
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
    from lmcache.v1.distributed.memory_manager import (
        L1MemoryManager,
        create_memory_allocator,
    )

    # ``_is_maru_allocator`` is a private helper not re-exported by the package
    # ``__init__``; import it from the submodule (else this whole module would
    # ImportError and silently skip).
    from lmcache.v1.distributed.memory_manager.l1_memory_manager import (
        _is_maru_allocator,
    )
except ImportError:
    pytest.skip(
        "MaruMemoryAllocator / memory_manager could not be imported",
        allow_module_level=True,
    )


@pytest.fixture
def maru_cfg() -> MaruL1Config:
    """Plausible MaruL1Config — the new lazy ``__init__`` performs no
    MaruServer RPC, so these values are not exercised unless a test
    explicitly drives ``init_layout``.
    """
    return MaruL1Config(
        server_url="maru://localhost:5555",
        pool_size_bytes=60 * 1024**3,
        instance_id="test-mp",
    )


# Tiny allocations so the dispatch tests don't pin gigabytes of host memory
# and starve subsequent ``MixedMemoryAllocator`` tests in the same process.
# ``LazyMemoryAllocator.__init__`` eagerly calls ``torch.empty(final_size)``
# and ``cudaHostRegister`` on ``init_size`` — so we keep both ≤ 1MB.
_TINY_BYTES = 1 << 20  # 1MB


# =========================================================================
# (1) L1MemoryManagerConfig — maru_config field
# =========================================================================


class TestL1MemoryManagerConfigMaru:
    def test_default_has_no_maru_config(self):
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=False)
        assert cfg.maru_config is None

    def test_default_clamps_init_size(self):
        # init_size_in_bytes defaults to 20GB; size_in_bytes=1MB → clamp to 1MB.
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=False)
        assert cfg.init_size_in_bytes == _TINY_BYTES

    def test_maru_config_skips_clamp(self, maru_cfg):
        # size_in_bytes=0 is OK when maru_config is set (DRAM fields ignored).
        # The default init_size_in_bytes (20GB) should NOT be clamped to 0.
        cfg = L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
        assert cfg.maru_config is maru_cfg
        assert cfg.init_size_in_bytes == 20 << 30


# =========================================================================
# (2) create_memory_allocator() dispatch
# =========================================================================


class TestCreateMemoryAllocatorDispatch:
    def test_lazy_path_unchanged(self):
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=True)
        alloc = create_memory_allocator(cfg)
        try:
            assert isinstance(alloc, LazyMemoryAllocator)
        finally:
            alloc.close()

    # NOTE: ``use_lazy=False`` (MixedMemoryAllocator) is intentionally
    # NOT covered here — its constructor eagerly invokes
    # ``cudaHostAlloc`` which is environment-dependent. That path is
    # already exercised by ``test_l1_memory_manager.py``; we only need
    # to verify the maru routing here.

    def test_maru_path_routes_to_maru_allocator(self, maru_cfg):
        cfg = L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
        alloc = create_memory_allocator(cfg)
        assert isinstance(alloc, MaruMemoryAllocator)
        # Lazy: handler / adapter are still ``None`` before init_layout.
        assert alloc._handler is None
        assert alloc._cxl_adapter is None
        assert alloc.is_initialized is False

    def test_maru_config_takes_precedence_over_use_lazy(self, maru_cfg):
        # use_lazy=True should be ignored when maru_config is set.
        cfg = L1MemoryManagerConfig(
            size_in_bytes=_TINY_BYTES, use_lazy=True, maru_config=maru_cfg
        )
        alloc = create_memory_allocator(cfg)
        assert isinstance(alloc, MaruMemoryAllocator)


# =========================================================================
# (3) _is_maru_allocator helper
# =========================================================================


class TestIsMaruAllocator:
    def test_returns_false_for_lazy(self):
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=True)
        alloc = create_memory_allocator(cfg)
        try:
            assert _is_maru_allocator(alloc) is False
        finally:
            alloc.close()

    def test_returns_true_for_maru(self, maru_cfg):
        cfg = L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
        alloc = create_memory_allocator(cfg)
        assert _is_maru_allocator(alloc) is True


# =========================================================================
# (4) L1MemoryManager.get_memory_usage() — maru case
# =========================================================================


def _make_maru_manager(maru_cfg) -> L1MemoryManager:
    """Build an ``L1MemoryManager`` whose allocator is a freshly
    constructed (uninitialized) maru allocator. Tests that exercise
    handler stats need to install ``_handler`` and ``_cxl_adapter``
    mocks on the allocator.
    """
    cfg = L1MemoryManagerConfig(size_in_bytes=0, use_lazy=False, maru_config=maru_cfg)
    return L1MemoryManager(cfg)


def _fake_init_layout(allocator: MaruMemoryAllocator) -> None:
    """Install ``MagicMock`` handler + adapter so the allocator
    behaves as ``is_initialized`` without contacting MaruServer.
    """
    allocator._handler = mock.MagicMock()
    allocator._cxl_adapter = mock.MagicMock()


class TestGetMemoryUsageMaru:
    def test_returns_zero_before_init_layout(self, maru_cfg):
        # Allocator constructed but ``init_layout`` not yet called.
        mgr = _make_maru_manager(maru_cfg)
        assert mgr.get_memory_usage() == (0, 0)

    def test_returns_zero_when_handler_has_no_get_stats(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr._allocator)
        # spec=[] → mock has no attributes (no ``get_stats``)
        mgr._allocator._handler = mock.Mock(spec=[])
        assert mgr.get_memory_usage() == (0, 0)

    def test_forwards_used_and_pool_size_bytes(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr._allocator)
        mgr._allocator._handler.get_stats.return_value = {
            "used_bytes": 1234,
            "pool_size_bytes": 5678,
        }
        assert mgr.get_memory_usage() == (1234, 5678)

    def test_falls_back_to_pool_size_key(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr._allocator)
        mgr._allocator._handler.get_stats.return_value = {
            "used_bytes": 100,
            "pool_size": 999,
        }
        assert mgr.get_memory_usage() == (100, 999)

    def test_returns_zero_on_handler_exception(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr._allocator)
        mgr._allocator._handler.get_stats.side_effect = RuntimeError("boom")
        # Should swallow and return (0, 0) rather than crash.
        assert mgr.get_memory_usage() == (0, 0)


# =========================================================================
# (5) L1MemoryManager.get_l1_memory_desc() — maru case
# =========================================================================


class TestGetL1MemoryDescMaru:
    def test_raises_not_implemented_for_maru(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        with pytest.raises(NotImplementedError, match="maru"):
            mgr.get_l1_memory_desc()


# =========================================================================
# (6) L1MemoryManager.register_kv_layout() — maru forwarding
# =========================================================================


class TestRegisterKvLayoutMaru:
    def test_forwards_to_allocator_init_layout(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        shapes = [torch.Size([2, 32, 256, 128])]
        dtypes = [torch.float16]
        with mock.patch.object(MaruMemoryAllocator, "init_layout") as mock_init_layout:
            mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256)
        mock_init_layout.assert_called_once_with(
            shapes, dtypes, MemoryFormat.KV_2LTD, 256
        )

    def test_default_backend_is_noop(self):
        # Lazy / Mixed backends are layout-agnostic — call must succeed
        # silently and not affect their internal state.
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=True)
        mgr = L1MemoryManager(cfg)
        try:
            mgr.register_kv_layout(
                [torch.Size([2, 32, 256, 128])],
                [torch.float16],
                MemoryFormat.KV_2LTD,
                256,
            )
        finally:
            mgr.close()
