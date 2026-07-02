# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the maru L1 allocator wiring.

In the sibling design, maru does NOT route through the shared
``L1MemoryManager`` (that stays maru-free). ``MaruL1Manager`` owns a
``MaruMemoryAllocator`` directly, wrapped by a small internal
``_MaruAllocatorMemoryManager`` that provides the ``allocate`` /
``get_memory_usage`` / ``register_kv_layout`` slice the dispatcher and
manager need.

Coverage:

1. ``L1MemoryManagerConfig.maru_config`` — when set, the DRAM-only
   ``init_size_in_bytes`` clamp is skipped.
2. ``MaruL1Manager.__init__`` — constructs a ``MaruMemoryAllocator``
   (lazy — no RPC before ``init_layout``).
3. ``MaruL1Manager.get_memory_usage()`` — best-effort forwarding to
   ``MaruHandler.get_stats``; short-circuits to ``(0, 0)`` before
   ``init_layout`` is called.
4. ``MaruL1Manager.get_l1_memory_desc()`` — returns ``None`` for maru
   (no contiguous DRAM buffer; copy-type-L2 contract).
5. ``MaruL1Manager.register_kv_layout()`` — forwards to
   ``MaruMemoryAllocator.init_layout``.

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
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.memory_management import MemoryFormat

try:
    # First Party
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


@pytest.fixture
def maru_cfg() -> MaruL1Config:
    """Plausible MaruL1Config — the lazy ``__init__`` performs no
    MaruServer RPC, so these values are not exercised unless a test
    explicitly drives ``init_layout``.
    """
    return MaruL1Config(
        server_url="maru://localhost:5555",
        pool_size_bytes=60 * 1024**3,
        instance_id="test-mp",
    )


# Tiny allocations so the dispatch tests don't pin gigabytes of host memory.
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
# (2) MaruL1Manager owns a MaruMemoryAllocator (lazy)
# =========================================================================


def _make_maru_manager(maru_cfg) -> MaruL1Manager:
    """Build a ``MaruL1Manager`` whose allocator is a freshly
    constructed (uninitialized) maru allocator. Tests that exercise
    handler stats need to install ``_handler`` / ``_cxl_adapter`` mocks
    on the allocator.
    """
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
    )
    return MaruL1Manager(cfg)


def _fake_init_layout(allocator: MaruMemoryAllocator) -> None:
    """Install ``MagicMock`` handler + adapter so the allocator
    behaves as ``is_initialized`` without contacting MaruServer.
    """
    allocator._handler = mock.MagicMock()
    allocator._cxl_adapter = mock.MagicMock()


class TestMaruManagerAllocator:
    def test_owns_maru_allocator(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        assert isinstance(mgr._allocator, MaruMemoryAllocator)
        # Lazy: handler / adapter are still ``None`` before init_layout.
        assert mgr._allocator._handler is None
        assert mgr._allocator._cxl_adapter is None
        assert mgr._allocator.is_initialized is False

    def test_missing_maru_config_raises(self):
        cfg = L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=_TINY_BYTES, use_lazy=False
            )
        )
        with pytest.raises(ValueError, match="maru_config"):
            MaruL1Manager(cfg)


# =========================================================================
# (3) MaruL1Manager.get_memory_usage() — maru case
# =========================================================================


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
# (4) MaruL1Manager.get_l1_memory_desc() — maru case
# =========================================================================


class TestGetL1MemoryDescMaru:
    def test_returns_none_for_maru(self, maru_cfg):
        # Copy-type L2 contract: no single contiguous buffer to register.
        mgr = _make_maru_manager(maru_cfg)
        assert mgr.get_l1_memory_desc() is None


# =========================================================================
# (5) MaruL1Manager.register_kv_layout() — maru forwarding
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
