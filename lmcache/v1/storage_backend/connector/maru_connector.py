# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import List, Optional, no_type_check
from urllib.parse import urlparse
import asyncio
import re

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def parse_size(size_str: str) -> int:
    """Parse human-readable size string (e.g., '1G', '100M', '1024K') to bytes."""
    if isinstance(size_str, int):
        return size_str
    s = str(size_str).strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$", s)
    if not match:
        try:
            return int(s)
        except ValueError:
            raise ValueError(
                f"Could not parse '{size_str}' as a size string or an integer."
            ) from None
    value, unit = float(match.group(1)), match.group(2)
    multipliers = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(value * multipliers.get(unit, 1))


@dataclass
class MaruConnectorConfig:
    """Configuration for Maru connector."""

    server_url: str = "tcp://localhost:5555"
    pool_size: int = 1024 * 1024 * 1024  # 1GB default
    instance_id: Optional[str] = None
    auto_connect: bool = True
    connection_timeout: float = 30.0
    operation_timeout: float = 10.0
    timeout_ms: int = 2000  # ZMQ socket timeout in milliseconds
    use_async_rpc: bool = True  # Use async DEALER-ROUTER RPC
    max_inflight: int = 64  # Max concurrent in-flight async requests
    eager_map: Optional[bool] = None  # None = defer to MaruConfig/env

    @staticmethod
    def from_url(url: str) -> "MaruConnectorConfig":
        """Parse maru://host:port to extract server address only."""
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5555
        server_url = f"tcp://{host}:{port}"

        return MaruConnectorConfig(
            server_url=server_url,
        )

    @staticmethod
    def from_lmcache_config(config: "LMCacheEngineConfig") -> "MaruConnectorConfig":
        """Load from extra_config dict.

        All Maru-specific settings should be configured here.
        Supports human-readable size strings (e.g., '4G', '500M')
        for maru_pool_size.
        """
        extra = config.extra_config or {}
        raw_pool_size = extra.get("maru_pool_size", 1024**3)
        pool_size = (
            parse_size(raw_pool_size)
            if isinstance(raw_pool_size, str)
            else int(raw_pool_size)
        )
        return MaruConnectorConfig(
            server_url=extra.get("maru_server_url", "tcp://localhost:5555"),
            pool_size=pool_size,
            instance_id=extra.get("maru_instance_id"),
            auto_connect=extra.get("maru_auto_connect", True),
            operation_timeout=float(extra.get("maru_operation_timeout", 10.0)),
            timeout_ms=int(extra.get("maru_timeout_ms", 2000)),
            use_async_rpc=extra.get("maru_use_async_rpc", True),
            max_inflight=int(extra.get("maru_max_inflight", 64)),
            eager_map=extra.get("maru_eager_map"),
        )


# Ping error codes
PING_SUCCESS = 0
PING_NOT_CONNECTED = 1
PING_RPC_ERROR = 2


class MaruConnector(RemoteConnector):
    """
    The remote url should start with "maru://" and have one host-port pair.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        logger.info("init MaruConnector")
        super().__init__(config, metadata)
        if config.use_layerwise:
            raise NotImplementedError(
                "Maru connector does not yet support layerwise KV cache."
            )

        self.url = url
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # extra_config for all settings, URL for server address only
        url_config = MaruConnectorConfig.from_url(url)
        if config.extra_config:
            self.maru_config = MaruConnectorConfig.from_lmcache_config(config)
            # Use URL-derived server_url unless explicitly overridden
            if not config.extra_config.get("maru_server_url"):
                self.maru_config.server_url = url_config.server_url
        else:
            self.maru_config = url_config

        logger.info(
            "Maru config: server_url=%s, pool_size=%d, instance_id=%s, eager_map=%s",
            self.maru_config.server_url,
            self.maru_config.pool_size,
            self.maru_config.instance_id,
            self.maru_config.eager_map,
        )

        # Initialize MaruHandler (lazy connection)
        self._handle = None
        self._connected = False

        # Metrics
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._connection_attempts = 0
        self._connection_failures = 0
        self._rpc_errors = 0

        # Try to connect if auto_connect is enabled
        if self.maru_config.auto_connect:
            self._init_handle()

    def _init_handle(self) -> bool:
        try:
            # Third Party
            from maru import MaruConfig, MaruHandler
        except ImportError:
            logger.error("maru package not installed. Install with: pip install maru")
            return False

        try:
            maru_cfg_kwargs = dict(
                server_url=self.maru_config.server_url,
                instance_id=self.maru_config.instance_id,
                pool_size=self.maru_config.pool_size,
                chunk_size_bytes=self.full_chunk_size_bytes,
                auto_connect=False,  # We'll connect manually
                timeout_ms=self.maru_config.timeout_ms,
                use_async_rpc=self.maru_config.use_async_rpc,
                max_inflight=self.maru_config.max_inflight,
            )
            if self.maru_config.eager_map is not None:
                maru_cfg_kwargs["eager_map"] = self.maru_config.eager_map
            maru_cfg = MaruConfig(**maru_cfg_kwargs)
            handle = MaruHandler(maru_cfg)
            self._handle = handle
            if handle.connect():
                self._connected = True
                self._connection_attempts += 1
                logger.info("init maru handler success")
                return True
            else:
                logger.error("fail to init maru handler, connect returned False")
                self._connection_attempts += 1
                self._connection_failures += 1
                self._handle = None
                return False
        except Exception as e:
            logger.error("fail to init maru handler: %s", e)
            self._connection_attempts += 1
            self._connection_failures += 1
            self._handle = None
            return False

    def _ensure_connected(self) -> bool:
        if self._connected and self._handle is not None:
            return True
        return self._init_handle()

    async def exists(self, key: CacheEngineKey) -> bool:
        if not self._ensure_connected():
            return False
        assert self._handle is not None

        key_str = key.to_string()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._handle.exists, key_str),
                timeout=self.maru_config.operation_timeout,
            )
            logger.debug(
                "maru exists key_str=%s, exists=%s",
                key_str,
                result,
            )
            return result
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning("maru exists timed out for key_str=%s", key_str)
            return False
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru exists failed: %s", e)
            return False

    def exists_sync(self, key: CacheEngineKey) -> bool:
        if not self._ensure_connected():
            return False
        assert self._handle is not None

        key_str = key.to_string()
        try:
            result = self._handle.exists(key_str)
            logger.debug("maru exists_sync key_str=%s, exists=%s", key_str, result)
            return result
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru exists_sync failed: %s", e)
            return False

    def _decode_memory_obj(self, info) -> Optional[MemoryObj]:
        mv = info.view

        logger.debug("maru decode data=%d bytes", len(mv))

        # memoryview -> torch tensor (zero-copy)
        raw_data = torch.frombuffer(mv, dtype=torch.uint8)

        meta = MemoryObjMetadata(
            shape=self.meta_shapes[0],
            dtype=self.meta_dtypes[0],
            address=0,
            phy_size=raw_data.numel(),
            ref_count=1,
            pin_count=0,
            fmt=self.meta_fmt,
            shapes=self.meta_shapes,
            dtypes=self.meta_dtypes,
        )

        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=meta,
            parent_allocator=None,
        )

    def _encode_memory_obj(self, memory_obj: MemoryObj):
        # Third Party
        from maru_handler.memory import MemoryInfo

        info = MemoryInfo(view=memory_obj.byte_array)
        logger.debug("maru encode data=%d bytes", len(info.view))
        return info

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if not self._ensure_connected():
            return None
        assert self._handle is not None

        key_str = key.to_string()
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(self._handle.retrieve, key_str),
                timeout=self.maru_config.operation_timeout,
            )
            if info is None:
                logger.debug("maru get MISS key_str=%s", key_str)
                return None

            data_size = len(info.view)
            logger.debug("maru get HIT key_str=%s, %d bytes", key_str, data_size)
            memory_obj = self._decode_memory_obj(info)
            if memory_obj is not None:
                memory_obj = self.reshape_partial_chunk(memory_obj, data_size)
            return memory_obj
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning("maru get timed out for key_str=%s", key_str)
            return None
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru get failed: %s", e)
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if not self._ensure_connected():
            raise RuntimeError("MaruConnector not connected to Maru server")
        assert self._handle is not None

        key_str = key.to_string()
        info = self._encode_memory_obj(memory_obj)
        data_size = len(info.view)

        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(self._handle.store, key_str, info),
                timeout=self.maru_config.operation_timeout,
            )
            if success:
                logger.debug("maru put key_str=%s, %d bytes", key_str, data_size)
            else:
                logger.warning("maru put failed key_str=%s", key_str)
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "maru put timed out for key_str=%s. Decode instance may redo prefill.",
                key_str,
            )
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru put failed: %s", e)
            raise

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    def remove_sync(self, key: CacheEngineKey) -> bool:
        if not self._ensure_connected():
            return False
        assert self._handle is not None

        key_str = key.to_string()
        try:
            return self._handle.delete(key_str)
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru remove_sync failed: %s", e)
            return False

    async def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception as e:
                logger.error("fail to close maru handler: %s", e)
            finally:
                self._handle = None
                self._connected = False
        logger.info("closed the maru connection")

    def support_batched_get(self) -> bool:
        return True

    def support_batched_put(self) -> bool:
        return True

    def support_batched_async_contains(self) -> bool:
        return True

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not self._ensure_connected() or not keys:
            return 0
        assert self._handle is not None

        key_strs = [k.to_string() for k in keys]
        try:
            results = self._handle.batch_exists(key_strs)
            count = 0
            for exists in results:
                if not exists:
                    break
                count += 1
            logger.debug("maru batched_contains hits=%d/%d", count, len(keys))
            return count
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru batched_contains failed: %s", e)
            return 0

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not self._ensure_connected() or not keys:
            return [None] * len(keys)
        assert self._handle is not None

        key_strs = [k.to_string() for k in keys]
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(self._handle.batch_retrieve, key_strs),
                timeout=self.maru_config.operation_timeout,
            )
            hits = sum(1 for r in raw_results if r is not None)
            logger.debug("maru batched_get hits=%d/%d", hits, len(keys))
            memory_objs: List[Optional[MemoryObj]] = []
            for info in raw_results:
                if info is None:
                    memory_objs.append(None)
                    continue
                memory_obj = self._decode_memory_obj(info)
                if memory_obj is not None:
                    memory_obj = self.reshape_partial_chunk(memory_obj, len(info.view))
                memory_objs.append(memory_obj)
            return memory_objs
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning("maru batched_get timed out for %d keys", len(keys))
            return [None] * len(keys)
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru batched_get failed: %s", e)
            return [None] * len(keys)

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ):
        if not self._ensure_connected() or not keys:
            return
        assert self._handle is not None

        key_strs = [k.to_string() for k in keys]
        infos = [self._encode_memory_obj(obj) for obj in memory_objs]
        total_bytes = sum(len(info.view) for info in infos)

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._handle.batch_store,
                    key_strs,
                    infos,
                ),
                timeout=self.maru_config.operation_timeout,
            )
            stored = sum(results) if results else 0
            if stored < len(keys):
                logger.warning(
                    "maru batched_put partial %d/%d keys",
                    stored,
                    len(keys),
                )
            else:
                logger.debug(
                    "maru batched_put %d keys, %d bytes",
                    len(keys),
                    total_bytes,
                )
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "maru batched_put timed out for %d keys. "
                "Decode instance may redo prefill.",
                len(keys),
            )
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru batched_put failed: %s", e)
            raise

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        if not self._ensure_connected() or not keys:
            return []
        assert self._handle is not None

        key_strs = [k.to_string() for k in keys]
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(self._handle.batch_retrieve, key_strs),
                timeout=self.maru_config.operation_timeout,
            )

            # Build consecutive prefix of hits
            memory_objs = []
            for info in raw_results:
                if info is None:
                    break
                memory_obj = self._decode_memory_obj(info)
                if memory_obj is None:
                    break
                memory_obj = self.reshape_partial_chunk(memory_obj, len(info.view))
                memory_objs.append(memory_obj)

            logger.debug(
                "maru batched_get_nb lookup_id=%s, hits=%d/%d",
                lookup_id,
                len(memory_objs),
                len(keys),
            )
            return memory_objs
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "maru batched_get_non_blocking timed out for lookup_id=%s, %d keys",
                lookup_id,
                len(keys),
            )
            return []
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru batched_get_non_blocking failed: %s", e)
            return []

    def support_batched_get_non_blocking(self) -> bool:
        return True

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        if not self._connected or self._handle is None:
            self._stats_monitor.update_remote_ping_error_code(PING_NOT_CONNECTED)
            return PING_NOT_CONNECTED
        try:
            healthy = await asyncio.wait_for(
                asyncio.to_thread(self._handle.healthcheck),
                timeout=self.maru_config.operation_timeout,
            )
            if not healthy:
                self._stats_monitor.update_remote_ping_error_code(PING_RPC_ERROR)
                return PING_RPC_ERROR
            self._stats_monitor.update_remote_ping_error_code(PING_SUCCESS)
            return PING_SUCCESS
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning("maru ping timed out")
            self._stats_monitor.update_remote_ping_error_code(PING_RPC_ERROR)
            return PING_RPC_ERROR
        except Exception as e:
            self._rpc_errors += 1
            logger.warning("maru ping failed: %s", e)
            self._stats_monitor.update_remote_ping_error_code(PING_RPC_ERROR)
            return PING_RPC_ERROR

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not self._ensure_connected() or not keys:
            return 0
        assert self._handle is not None

        key_strs = [k.to_string() for k in keys]
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(self._handle.batch_exists, key_strs),
                timeout=self.maru_config.operation_timeout,
            )
            # Count consecutive hits from start
            count = 0
            for exists in results:
                if not exists:
                    break
                count += 1
            logger.debug(
                "maru batched_async_contains lookup_id=%s, hits=%d/%d",
                lookup_id,
                count,
                len(keys),
            )
            return count
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "maru batched_async_contains timed out for lookup_id=%s, %d keys",
                lookup_id,
                len(keys),
            )
            return 0
        except Exception as e:
            self._rpc_errors += 1
            logger.error("maru batched_async_contains failed: %s", e)
            return 0
