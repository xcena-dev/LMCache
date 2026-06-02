# SPDX-License-Identifier: Apache-2.0
"""Seam A draft — allocator polymorphism replacing the gpu_ops isinstance chain.

This is a REVIEW DRAFT. It is not imported anywhere.

Goal: `lmcache/v1/gpu_connector/gpu_ops.py` should not know about any concrete
allocator type. Today it does:

    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        ...lazy path...
    else:
        ...default cudaMemcpy path...

and PR #3420 adds a third `isinstance(parent, GdsScratchAllocator)` branch.
Instead, each allocator owns its own H2D/D2H transfer behind two new methods on
`MemoryAllocatorInterface`. Existing behavior is byte-for-byte preserved.
"""

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
import lmcache.c_ops as lmc_ops


# ---------------------------------------------------------------------------
# 1) New polymorphic methods on MemoryAllocatorInterface
#    (added to lmcache/v1/memory_management.py, class at line ~829)
# ---------------------------------------------------------------------------
#
# Add these two NON-abstract methods so every existing allocator inherits the
# default DRAM behavior and only special media override them.
#
class _MemoryAllocatorInterface_additions:  # illustrative mixin, not real
    def copy_to_gpu(
        self,
        memory_obj: "MemoryObj",
        gpu_buffer: torch.Tensor,
    ) -> None:
        """Copy ``memory_obj`` into ``gpu_buffer`` (H2D), stream-ordered.

        Default implementation: a non-blocking ``cudaMemcpyAsync`` from the
        object's host tensor. This is the path used by every DRAM-backed
        allocator today. Non-DRAM media override this.

        Args:
            memory_obj: Source object owned by this allocator. Its
                ``raw_tensor`` must be allocated.
            gpu_buffer: Destination GPU buffer; ``nbytes`` must equal
                ``memory_obj.get_size()`` (checked by the caller).

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

    def copy_from_gpu(
        self,
        gpu_buffer: torch.Tensor,
        memory_obj: "MemoryObj",
    ) -> None:
        """Copy ``gpu_buffer`` into ``memory_obj`` (D2H), stream-ordered.

        Default implementation mirrors :meth:`copy_to_gpu`. Non-DRAM media
        override this.

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
# 2) LazyMemoryAllocator override
#    (added to lmcache/v1/lazy_memory_allocator.py)
# ---------------------------------------------------------------------------
class _LazyMemoryAllocator_additions:  # illustrative, not real
    def copy_to_gpu(self, memory_obj, gpu_buffer) -> None:
        size = memory_obj.get_size()
        lmc_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            size,
            lmc_ops.TransferDirection.H2D,
            memory_obj.meta.address,
            self.PIN_CHUNK_SIZE,
        )

    def copy_from_gpu(self, gpu_buffer, memory_obj) -> None:
        size = memory_obj.get_size()
        lmc_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            size,
            lmc_ops.TransferDirection.D2H,
            memory_obj.meta.address,
            self.PIN_CHUNK_SIZE,
        )


# ---------------------------------------------------------------------------
# 3) GdsScratchAllocator override (PR #3420)
#    (the existing cufile_read_into / cufile_write_from, just renamed to the
#     interface methods — or thin wrappers calling them)
# ---------------------------------------------------------------------------
class _GdsScratchAllocator_additions:  # illustrative, not real
    def copy_to_gpu(self, memory_obj, gpu_buffer) -> None:
        # NVMe -> registered VRAM via cuFile DMA.
        self.cufile_read_into(memory_obj, gpu_buffer)

    def copy_from_gpu(self, gpu_buffer, memory_obj) -> None:
        # registered VRAM -> NVMe via cuFile DMA.
        self.cufile_write_from(memory_obj, gpu_buffer)


# ---------------------------------------------------------------------------
# 4) The new gpu_ops.py — no concrete allocator imports, no isinstance
# ---------------------------------------------------------------------------
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
) -> None:
    """Copy a MemoryObj to a GPU buffer, dispatching on its owning allocator.

    Non-blocking; no stream synchronization. The actual transfer mechanism
    (cudaMemcpyAsync, lazy-pin, cuFile DMA, ...) is chosen polymorphically by
    the object's parent allocator.
    """
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().copy_to_gpu(memory_obj, gpu_buffer)


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
) -> None:
    """Copy a GPU buffer into a MemoryObj, dispatching on its owning allocator."""
    _check_size(memory_obj, gpu_buffer)
    memory_obj.parent().copy_from_gpu(gpu_buffer, memory_obj)


def _check_size(memory_obj: MemoryObj, gpu_buffer: torch.Tensor) -> None:
    """Validate that the GPU buffer matches the MemoryObj payload size.

    Raises:
        ValueError: On a size mismatch.
    """
    size = memory_obj.get_size()
    if size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
