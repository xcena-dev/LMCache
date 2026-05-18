# SPDX-License-Identifier: Apache-2.0

"""Maru-backed L1 memory allocator for MP mode.

This module exposes :class:`MaruMemoryAllocator`, an implementation of
:class:`MemoryAllocatorInterface` whose ``MemoryObj`` instances are backed
by CXL shared memory via the embedded ``CxlMemoryAdapter``
(``maru_lmcache``). It is the "Option B" allocator used by LMCache MP
mode's maru integration; see ``docs/source/mp/maru/integration.md`` for
the broader design.

Key invariants:
- ``MemoryObj.parent_allocator`` is ``None`` for all objects returned by
  this allocator. LMCache's refcount-driven free path must NOT release
  the underlying CXL pages — lifecycle is owned by ``MaruServer``
  (``pin_kv`` / ``unpin_kv`` / ``delete_kv``).
- :meth:`get_by_location` and :meth:`create_store_handle` are not part
  of ``MemoryAllocatorInterface``; ``L1Manager``'s maru branch reaches
  them through an ``isinstance`` check.
- ``maru`` and ``maru_lmcache`` are imported lazily so that loading this
  module does not require those packages to be installed.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, List, Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)


@dataclass
class MaruL1Config:
    """Configuration for :class:`MaruMemoryAllocator`.

    Attributes:
        server_url: MaruServer endpoint. Accepts both ``maru://host:port``
            and ``tcp://host:port``; the former is rewritten to the
            latter internally.
        pool_size_bytes: Per-instance CXL pool quota requested from
            ``MaruServer``.
        full_chunk_size_bytes: Total bytes of a single full KV chunk
            (across layers / tokens / hidden dims). Must be a multiple
            of ``chunk_size_in_tokens``.
        chunk_size_in_tokens: LMCache chunk size in tokens (e.g. 256).
            Used together with ``full_chunk_size_bytes`` to derive
            ``single_token_size`` for partial-chunk handling in
            :meth:`MaruMemoryAllocator.get_by_location`.
        shapes: KV chunk shapes (forwarded to ``CxlMemoryAdapter``).
        dtypes: KV chunk dtypes (forwarded to ``CxlMemoryAdapter``).
        fmt: Memory format (e.g. ``KV_2LTD`` or ``KV_MLA_FMT``).
        instance_id: Stable identifier for this client instance, used by
            ``MaruServer`` for ownership tracking, restart recovery, and
            observability. If ``None``, ``MaruConfig`` auto-generates a
            UUID (acceptable for single-instance / single-node setups but
            not recommended for multi-node deployments).
        timeout_ms: Socket timeout for RPC calls.
        use_async_rpc: Whether to use async DEALER-ROUTER RPC client.
        max_inflight: Max concurrent in-flight async requests.
        eager_map: Pre-map all shared regions on connect.
    """

    server_url: str
    pool_size_bytes: int
    full_chunk_size_bytes: int
    chunk_size_in_tokens: int
    shapes: List[torch.Size]
    dtypes: List[torch.dtype]
    fmt: MemoryFormat
    instance_id: Optional[str] = None
    timeout_ms: int = 5000
    use_async_rpc: bool = True
    max_inflight: int = 64
    eager_map: bool = True


class MaruMemoryAllocator(MemoryAllocatorInterface):
    """L1 memory allocator backed by CXL shared memory via Maru.

    Wraps the embedded ``CxlMemoryAdapter`` (``maru_lmcache``) and
    re-exposes the slice of its API required by
    :class:`MemoryAllocatorInterface`. Adds :meth:`get_by_location` and
    :meth:`create_store_handle` for callers — chiefly ``L1Manager``'s
    maru branch — that need to drive ``MaruHandler`` RPCs directly.

    The constructor connects to ``MaruServer`` eagerly. Failure to
    connect raises ``RuntimeError`` with no retry; higher layers are
    expected to fall back to a non-maru backend if this happens.
    """

    def __init__(self, config: MaruL1Config) -> None:
        # Lazy import so this module can be imported without the maru
        # runtime present (e.g. in non-maru deployments).
        # Third Party
        from maru import MaruConfig, MaruHandler
        from maru_lmcache import CxlMemoryAdapter

        if config.full_chunk_size_bytes <= 0:
            raise ValueError(
                f"full_chunk_size_bytes must be positive, "
                f"got {config.full_chunk_size_bytes}"
            )
        if config.chunk_size_in_tokens <= 0:
            raise ValueError(
                f"chunk_size_in_tokens must be positive, "
                f"got {config.chunk_size_in_tokens}"
            )
        if config.full_chunk_size_bytes % config.chunk_size_in_tokens != 0:
            raise ValueError(
                f"full_chunk_size_bytes ({config.full_chunk_size_bytes}) must be "
                f"a multiple of chunk_size_in_tokens "
                f"({config.chunk_size_in_tokens})"
            )

        self._config = config
        self._single_token_size: int = (
            config.full_chunk_size_bytes // config.chunk_size_in_tokens
        )

        # ``MaruHandler`` expects the ``tcp://`` scheme; ``maru://`` is
        # the LMCache-facing convention (mirrors ``MaruBackend``).
        server_url = config.server_url
        if server_url.startswith("maru://"):
            server_url = "tcp://" + server_url[len("maru://") :]

        maru_config = MaruConfig(
            server_url=server_url,
            instance_id=config.instance_id,
            pool_size=config.pool_size_bytes,
            chunk_size_bytes=config.full_chunk_size_bytes,
            auto_connect=False,
            timeout_ms=config.timeout_ms,
            use_async_rpc=config.use_async_rpc,
            max_inflight=config.max_inflight,
            eager_map=config.eager_map,
        )

        self._handler: Any = MaruHandler(maru_config)
        if not self._handler.connect():
            raise RuntimeError(f"Failed to connect MaruHandler to {config.server_url}")
        logger.info(
            "[MaruMemoryAllocator] connected: server=%s instance_id=%s "
            "pool_size=%d chunk_size_bytes=%d",
            config.server_url,
            self._handler.instance_id,
            config.pool_size_bytes,
            config.full_chunk_size_bytes,
        )

        self._cxl_adapter: Any = CxlMemoryAdapter(
            handler=self._handler,
            shapes=config.shapes,
            dtypes=config.dtypes,
            fmt=config.fmt,
            chunk_size=self._handler.get_chunk_size(),
        )

    # ------------------------------------------------------------------
    # Accessors used by L1Manager's maru branch (via isinstance check)
    # ------------------------------------------------------------------

    @property
    def handler(self) -> Any:
        """The connected ``MaruHandler``.

        ``L1Manager``'s maru branch uses this to issue ``batch_store`` /
        ``batch_pin`` / ``batch_retrieve`` / ``batch_unpin`` / ``delete``
        directly, bypassing the ``L2AdapterInterface`` framework.
        """
        return self._handler

    @property
    def single_token_size(self) -> int:
        """Bytes per single token in a KV chunk.

        Used by ``L1Manager``'s maru branch when invoking
        :meth:`get_by_location` to materialize a ``MemoryObj`` for a
        partial chunk.
        """
        return self._single_token_size

    def get_by_location(
        self,
        region_id: int,
        page_index: int,
        actual_size: int,
        single_token_size: Optional[int] = None,
    ) -> Optional[MemoryObj]:
        """Resolve a CXL ``(region_id, page_index)`` to a ``MemoryObj``.

        Used during the RETRIEVE lookup phase: ``MaruHandler.batch_retrieve``
        reports the location and this method materialises the
        pool-resident ``MemoryObj`` (no data copy).

        Args:
            region_id: Region id from ``MaruServer``.
            page_index: Page index within the region.
            actual_size: Actual KV chunk size in bytes (may be less than
                a full chunk for trailing partial chunks).
            single_token_size: Bytes-per-token override for partial
                chunks. Defaults to :attr:`single_token_size`.

        Returns:
            ``MemoryObj`` resolving the CXL page, or ``None`` if the
            location is no longer valid (e.g. region not mapped).
        """
        if single_token_size is None:
            single_token_size = self._single_token_size
        return self._cxl_adapter.get_by_location(
            region_id=region_id,
            page_index=page_index,
            actual_size=actual_size,
            single_token_size=single_token_size,
        )

    def create_store_handle(self, memory_obj: MemoryObj) -> Any:
        """Reconstruct an ``AllocHandle`` from a ``MemoryObj`` for use
        with ``MaruHandler.batch_store``.
        """
        return self._cxl_adapter.create_store_handle(memory_obj)

    # ------------------------------------------------------------------
    # MemoryAllocatorInterface
    # ------------------------------------------------------------------

    def allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate a single CXL-backed ``MemoryObj``.

        ``CxlMemoryAdapter`` uses the canonical shapes/dtypes/fmt fixed
        at construction time; the arguments here are accepted for
        interface compatibility but the pool's metadata is authoritative.
        """
        return self._cxl_adapter.allocate(shapes, dtypes, fmt, allocator_type)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate ``batch_size`` CXL-backed ``MemoryObj`` instances."""
        return self._cxl_adapter.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """No-op. CXL lifecycle owned by MaruServer."""
        return

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """No-op. CXL lifecycle owned by MaruServer."""
        return

    def close(self) -> None:
        """Close the underlying ``CxlMemoryAdapter`` and ``MaruHandler``.

        Best-effort: errors during close are logged but do not propagate
        to avoid masking earlier shutdown failures.
        """
        try:
            self._cxl_adapter.close()
        except Exception:
            logger.exception("[MaruMemoryAllocator] CxlMemoryAdapter.close() failed")
        try:
            self._handler.close()
        except Exception:
            logger.exception("[MaruMemoryAllocator] MaruHandler.close() failed")

    def memcheck(self) -> bool:
        return True
