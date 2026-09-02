# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_gds_context
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.platform.ops_types import StagingCopy
import lmcache.lmcache_native as lmcache_native


# Helper functions
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
):
    """Helper function to copy memory object allocated by different
    allocators to GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param MemoryObj memory_obj: The memory object to be copied.
    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data to.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        get_gds_context().transfer_async(memory_obj, gpu_buffer, SlabDirection.READ)
        return
    src_tensor = memory_obj.raw_tensor
    if src_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        device_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.data_ptr,
            mem_obj_size,
            lmcache_native.TransferDirection.H2D,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        gpu_buffer.view(torch.uint8).copy_(
            src_tensor.view(torch.uint8)[:mem_obj_size], non_blocking=True
        )


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
):
    """Helper function to copy memory object allocated by different
    allocators from GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        get_gds_context().transfer_async(memory_obj, gpu_buffer, SlabDirection.WRITE)
        return
    dst_tensor = memory_obj.raw_tensor
    if dst_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    mem_obj_size = memory_obj.get_size()
    if mem_obj_size != gpu_buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
            f"gpu_buffer nbytes={gpu_buffer.nbytes}"
        )
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        device_ops.lmcache_memcpy_async(
            memory_obj.data_ptr,
            gpu_buffer.data_ptr(),
            mem_obj_size,
            lmcache_native.TransferDirection.D2H,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        dst_tensor.view(torch.uint8)[:mem_obj_size].copy_(
            gpu_buffer.view(torch.uint8), non_blocking=True
        )


def build_layer_staging_copies(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
    *,
    kv_size: int,
    num_layers: int,
    layer_start: int,
    layer_count: int,
) -> list[StagingCopy]:
    """Build ``StagingCopy`` descriptors for one layer slice of a chunk batch.

    Layer-major counterpart of :func:`build_staging_copies`. The host object
    keeps its ``[kv_size, num_layers, tokens, hidden]`` layout; a layer slice is
    therefore ``kv_size`` contiguous byte ranges inside it (two for MHA, one for
    MLA), not one. The staged GPU buffer holds the slice packed as
    ``[kv_size, layer_count, tokens, hidden]``, which is what the scatter kernel
    addresses when it is given ``staged_layers=layer_count``.

    Args:
        memory_objs: Lazy-allocator memory objects, one per chunk in the batch.
        gpu_buffers: GPU staging buffers, aligned element-wise with
            ``memory_objs``. Each must be exactly the slice size.
        is_h2d: True for retrieve (CPU->GPU), False for store (GPU->CPU).
        kv_size: 1 (MLA) or 2 (separate K and V planes).
        num_layers: Layers in the host object (the model's layer count).
        layer_start: First layer of the slice.
        layer_count: Layers in the slice.

    Returns:
        ``kv_size`` descriptors per object, K plane before V plane, objects in
        input order.

    Raises:
        ValueError: If an object has not been allocated, if the slice falls
            outside ``num_layers``, or if a GPU buffer is not the slice size.
    """
    if kv_size < 1:
        raise ValueError(f"kv_size must be >= 1, got {kv_size}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if layer_count < 1:
        raise ValueError(f"layer_count must be >= 1, got {layer_count}")
    if layer_start < 0 or layer_start + layer_count > num_layers:
        raise ValueError(
            f"layer slice [{layer_start}, {layer_start + layer_count}) is "
            f"outside the object's {num_layers} layers"
        )

    copies: list[StagingCopy] = []
    for memory_obj, gpu_buffer in zip(memory_objs, gpu_buffers, strict=True):
        if memory_obj.raw_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj has been "
                "allocated."
            )
        mem_obj_size = memory_obj.get_size()
        if mem_obj_size % (kv_size * num_layers) != 0:
            raise ValueError(
                f"memory_obj nbytes={mem_obj_size} is not divisible by "
                f"kv_size*num_layers={kv_size * num_layers}; the object is not "
                "in the expected [kv, layer, token, hidden] layout"
            )
        plane_bytes = mem_obj_size // kv_size
        layer_bytes = plane_bytes // num_layers
        slice_bytes = layer_bytes * layer_count
        expected = slice_bytes * kv_size
        if gpu_buffer.nbytes != expected:
            raise ValueError(
                f"Size mismatch: gpu_buffer nbytes={gpu_buffer.nbytes}, "
                f"layer-slice nbytes={expected}"
            )

        host_base = memory_obj.data_ptr
        host_offset_base = memory_obj.meta.address
        gpu_base = gpu_buffer.data_ptr()
        for k_or_v in range(kv_size):
            # Host: plane k_or_v, layers [layer_start, layer_start+count).
            # Staged: plane k_or_v of a layer_count-deep buffer, so the planes
            # sit slice_bytes apart rather than plane_bytes apart.
            host_delta = k_or_v * plane_bytes + layer_start * layer_bytes
            gpu_delta = k_or_v * slice_bytes
            host_ptr = host_base + host_delta
            gpu_ptr = gpu_base + gpu_delta
            # host_offset drives the native memcpy's host-pin chunk splitting,
            # so it has to advance with the range we actually read/write.
            host_offset = host_offset_base + host_delta
            if is_h2d:
                copies.append(
                    device_ops.StagingCopy(gpu_ptr, host_ptr, slice_bytes, host_offset)
                )
            else:
                copies.append(
                    device_ops.StagingCopy(host_ptr, gpu_ptr, slice_bytes, host_offset)
                )
    return copies


def build_staging_copies(
    memory_objs: Sequence[MemoryObj],
    gpu_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list[StagingCopy]:
    """Build native ``StagingCopy`` descriptors for one batch of lazy objects.

    The H2D/D2H direction decides which side is source vs. destination; the host
    side is always the lazy memory object. Callers must ensure every object is
    lazy-allocator-backed.

    Args:
        memory_objs: Lazy-allocator memory objects, one per chunk in the batch.
        gpu_buffers: GPU staging buffers, aligned element-wise with
            ``memory_objs``.
        is_h2d: True for retrieve (CPU->GPU), False for store (GPU->CPU).

    Returns:
        One ``device_ops.StagingCopy`` per object, in input order.

    Raises:
        ValueError: If an object has not been allocated (``raw_tensor`` is None)
            or its size does not match its GPU buffer.
    """
    copies: list[StagingCopy] = []
    for memory_obj, gpu_buffer in zip(memory_objs, gpu_buffers, strict=True):
        if memory_obj.raw_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj has been "
                "allocated."
            )
        mem_obj_size = memory_obj.get_size()
        if mem_obj_size != gpu_buffer.nbytes:
            raise ValueError(
                f"Size mismatch: memory_obj nbytes={mem_obj_size}, "
                f"gpu_buffer nbytes={gpu_buffer.nbytes}"
            )
        host_ptr = memory_obj.data_ptr
        gpu_ptr = gpu_buffer.data_ptr()
        host_offset = memory_obj.meta.address
        if is_h2d:
            copies.append(
                device_ops.StagingCopy(gpu_ptr, host_ptr, mem_obj_size, host_offset)
            )
        else:
            copies.append(
                device_ops.StagingCopy(host_ptr, gpu_ptr, mem_obj_size, host_offset)
            )
    return copies
