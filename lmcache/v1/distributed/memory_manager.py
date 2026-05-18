# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.maru_config is not None:
        # Maru backend — CXL-backed allocator via MaruMemoryAllocator.
        # Lazy import keeps the maru runtime optional for non-maru builds.
        # First Party
        from lmcache.v1.distributed.maru_memory_allocator import MaruMemoryAllocator

        logger.debug(
            "use maru memory allocator: server=%s pool_size=%d bytes",
            config.maru_config.server_url,
            config.maru_config.pool_size_bytes,
        )
        return MaruMemoryAllocator(config.maru_config)
    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
        )


def _is_maru_allocator(allocator: MemoryAllocatorInterface) -> bool:
    """``isinstance(allocator, MaruMemoryAllocator)`` with lazy import.

    Avoids importing the maru-backed allocator (and indirectly the maru
    runtime types it lazily uses) when not needed.
    """
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import MaruMemoryAllocator

    return isinstance(allocator, MaruMemoryAllocator)


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    @property
    def allocator(self) -> MemoryAllocatorInterface:
        """Underlying memory allocator.

        Exposed primarily for callers that need allocator-specific
        operations not in :class:`MemoryAllocatorInterface` — e.g.
        ``L1Manager``'s maru branch reaches into
        :class:`MaruMemoryAllocator` for ``handler`` /
        ``get_by_location`` / ``create_store_handle``.
        """
        return self._allocator

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        self._allocator.batched_free(mem_objs)
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """
        # Maru backend: query MaruHandler stats. Eviction is owned by
        # MaruServer so this is best-effort observability; on failure
        # return (0, 0) rather than crash the eviction controller.
        if _is_maru_allocator(self._allocator):
            allocator = self._allocator
            # Lazy backend — handler not built until register_kv_layout.
            if not allocator.is_initialized:  # type: ignore[attr-defined]
                return 0, 0
            try:
                handler = allocator.handler  # type: ignore[attr-defined]
                stats = handler.get_stats() if hasattr(handler, "get_stats") else {}
                used = int(stats.get("used_bytes", 0))
                total = int(
                    stats.get("pool_size_bytes", 0) or stats.get("pool_size", 0)
                )
                return used, total
            except Exception:
                logger.exception("Failed to query Maru handler stats")
                return 0, 0

        # HACK: now trying to read this from the address manager in a ad-hoc
        # manner
        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            elif isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            else:
                raise NotImplementedError(
                    "get_memory_usage is not implemented for this allocator type."
                )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            L1MemoryDesc: Pointer, size, and alignment of the L1 buffer.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if _is_maru_allocator(self._allocator):
            # No contiguous DRAM buffer to describe — Maru-backed L1 lives in
            # CXL pages mmap'd via the handler. RDMA-style registration of a
            # single base pointer does not apply.
            raise NotImplementedError(
                "get_l1_memory_desc is not supported for the maru backend "
                "(L1 lives in CXL via mmap, not a single contiguous buffer)."
            )
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout to the underlying allocator.

        Only the maru backend acts on this — its ``CxlMemoryAdapter``
        pool is typed at first registration. The default DRAM
        allocators (``LazyMemoryAllocator`` / ``MixedMemoryAllocator``)
        are layout-agnostic so this call is a no-op for them.

        Idempotent for matching layouts; layout mismatch on a
        subsequent call raises ``ValueError`` (maru single-model
        constraint).

        Args:
            shapes: KV chunk shapes (per-layer-group).
            dtypes: KV chunk dtypes aligned with ``shapes``.
            fmt: Memory format.
            chunk_size_in_tokens: LMCache chunk size in tokens.
        """
        if _is_maru_allocator(self._allocator):
            self._allocator.init_layout(  # type: ignore[attr-defined]
                shapes, dtypes, fmt, chunk_size_in_tokens
            )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
