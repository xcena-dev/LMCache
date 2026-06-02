# SPDX-License-Identifier: Apache-2.0
"""Integration point 1 (the DMA) draft — allocator owns its GPU DMA.

REVIEW DRAFT. Not imported anywhere.

L1 ≡ GPU-DMA-able memory (issue #3262). The single operation the core needs from
any L1 backend is "DMA your bytes into / out of a GPU buffer". Today
`gpu_ops.py` hard-codes that per medium with an `isinstance` chain
(LazyMemoryAllocator today; PR #3420 adds a GdsScratchAllocator branch).

Make it a method the allocator owns. The default is the DRAM cudaMemcpyAsync
path; non-DRAM L1 backends override with their native DMA. `gpu_ops` then knows
no concrete type. Existing behavior is byte-for-byte preserved.
"""

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
import lmcache.c_ops as lmc_ops


# ---------------------------------------------------------------------------
# 1) New methods on MemoryAllocatorInterface
#    (lmcache/v1/memory_management.py, class at line ~829)
# ---------------------------------------------------------------------------
class _MemoryAllocatorInterface_additions:  # illustrative, not real
    def dma_to_gpu(
        self,
        memory_obj: "MemoryObj",
        gpu_buffer: torch.Tensor,
    ) -> None:
        """DMA ``memory_obj``'s bytes INTO ``gpu_buffer`` (load / H2D).

        This is the operation that defines L1: a real DMA into VRAM with no
        host staging. The default is a stream-ordered ``cudaMemcpyAsync`` from
        the object's host tensor (the DRAM L1 path). A non-DRAM L1 backend
        overrides this with its native DMA (cuFile for GDS, etc.).

        Non-blocking; no stream synchronization.

        Args:
            memory_obj: Source object owned by this allocator; ``raw_tensor``
                must be allocated.
            gpu_buffer: Destination GPU buffer; ``nbytes`` must equal
                ``memory_obj.get_size()`` (validated by the caller).

        Raises:
            ValueError: If ``memory_obj.raw_tensor`` is None.
        """
        src_tensor = memory_obj.raw_tensor
        if src_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj "
                "has been allocated."
            )
        size = memory_obj.get_size()
        gpu_buffer.view(torch.uint8).copy_(
            src_tensor.view(torch.uint8)[:size], non_blocking=True
        )

    def dma_from_gpu(
        self,
        gpu_buffer: torch.Tensor,
        memory_obj: "MemoryObj",
    ) -> None:
        """DMA ``gpu_buffer``'s bytes INTO ``memory_obj`` (evict / D2H).

        Default mirrors :meth:`dma_to_gpu`. Non-DRAM L1 backends override.

        Raises:
            ValueError: If ``memory_obj.raw_tensor`` is None.
        """
        dst_tensor = memory_obj.raw_tensor
        if dst_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj "
                "has been allocated."
            )
        size = memory_obj.get_size()
        dst_tensor.view(torch.uint8)[:size].copy_(
            gpu_buffer.view(torch.uint8), non_blocking=True
        )


# ---------------------------------------------------------------------------
# 2) LazyMemoryAllocator override (lmcache/v1/lazy_memory_allocator.py)
# ---------------------------------------------------------------------------
class _LazyMemoryAllocator_additions:  # illustrative, not real
    def dma_to_gpu(self, memory_obj, gpu_buffer) -> None:
        lmc_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            memory_obj.get_size(),
            lmc_ops.TransferDirection.H2D,
            memory_obj.meta.address,
            self.PIN_CHUNK_SIZE,
        )

    def dma_from_gpu(self, gpu_buffer, memory_obj) -> None:
        lmc_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            memory_obj.get_size(),
            lmc_ops.TransferDirection.D2H,
            memory_obj.meta.address,
            self.PIN_CHUNK_SIZE,
        )


# ---------------------------------------------------------------------------
# 3) GdsScratchAllocator override (PR #3420) — the cuFile P2P DMA
# ---------------------------------------------------------------------------
class _GdsScratchAllocator_additions:  # illustrative, not real
    def dma_to_gpu(self, memory_obj, gpu_buffer) -> None:
        # NVMe -> registered VRAM via cuFile P2P DMA (no host staging).
        self.cufile_read_into(memory_obj, gpu_buffer)

    def dma_from_gpu(self, gpu_buffer, memory_obj) -> None:
        # registered VRAM -> NVMe via cuFile P2P DMA.
        self.cufile_write_from(memory_obj, gpu_buffer)


# ---------------------------------------------------------------------------
# 4) The new gpu_ops.py — no concrete allocator imports, no isinstance
# ---------------------------------------------------------------------------
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
) -> None:
    """DMA a MemoryObj into a GPU buffer; the owning allocator picks the DMA.

    Non-blocking; no stream synchronization.
    """
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().dma_to_gpu(memory_obj, gpu_buffer)


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
) -> None:
    """DMA a GPU buffer into a MemoryObj; the owning allocator picks the DMA."""
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().dma_from_gpu(gpu_buffer, memory_obj)


def _check_size(memory_obj: MemoryObj, gpu_buffer: torch.Tensor) -> None:
    """Validate the GPU buffer matches the MemoryObj payload size.

    Raises:
        ValueError: On a size mismatch.
    """
    size = memory_obj.get_size()
    if size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
