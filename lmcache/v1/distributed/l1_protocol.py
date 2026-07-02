# SPDX-License-Identifier: Apache-2.0
"""Structural interface shared by the L1 manager control-plane tiers.

This is the *control-plane* seam: it captures the public method surface
that ``StorageManager`` and the storage controllers invoke on their L1
manager. Both the default
:class:`~lmcache.v1.distributed.l1_manager.L1Manager` and the maru
sibling :class:`~lmcache.v1.distributed.maru_l1_manager.MaruL1Manager`
satisfy it structurally, so ``StorageManager`` can hold either behind one
type.

This mirrors the GDS pattern in
``memory_manager/l1_manager_protocol.py`` (``L1ManagerProtocol``), but at
a different layer: that protocol is the *allocator / memory-manager*
seam (``allocate`` / ``free`` / ``get_l1_memory_desc`` …), whereas this
one is the *control* seam (``reserve_write`` / ``finish_write`` /
``reserve_read`` … — the pin/register/reserve/finish operations). The two
are intentionally distinct and must not be confused.

Because it is a structural :class:`typing.Protocol`, ``L1Manager`` needs
zero changes to satisfy it: it neither imports nor references this
module. ``l1_manager.py`` stays byte-identical to ``dev``.
"""

# Standard
from typing import Literal, Optional, Protocol, runtime_checkable

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1MemoryDesc,
    L1OperationResult,
)
from lmcache.v1.memory_management import MemoryFormat


@runtime_checkable
class L1ManagerInterface(Protocol):
    """Structural control-plane interface for an L1 manager.

    Signatures are derived from the public methods of the default
    :class:`L1Manager`. ``register_kv_layout`` is maru-only (default
    ``L1Manager`` does not define it), so it is intentionally omitted
    here; ``StorageManager`` gates its use behind the maru branch.
    """

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve write access (allocate) for the given keys."""
        ...

    def finish_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys."""
        ...

    def reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys."""
        ...

    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Read read-locked objects without acquiring new read locks."""
        ...

    def finish_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys."""
        ...

    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire a read lock."""
        ...

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache."""
        ...

    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache."""
        ...

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Mark the given keys as accessed (retrieved or stored)."""
        ...

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events."""
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)``."""
        ...

    def get_l1_memory_desc(self) -> Optional[L1MemoryDesc]:
        """Describe the underlying L1 buffer for L2-adapter registration.

        Returns ``None`` for tiers with no single registerable L1 buffer
        (e.g. maru — L1 lives in per-region CXL mmaps).
        """
        ...

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Whether the key is eligible for eviction (not locked)."""
        ...

    def get_object_state(self, key: ObjectKey) -> object:
        """Return the internal object state, or ``None`` if absent."""
        ...

    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state."""
        ...

    def memcheck(self) -> bool:
        """Perform a memory-consistency check for L1 cache."""
        ...

    def close(self) -> None:
        """Close the L1 manager and release all resources."""
        ...


@runtime_checkable
class MaruL1ManagerInterface(L1ManagerInterface, Protocol):
    """Maru extension of :class:`L1ManagerInterface`.

    Adds ``register_kv_layout``, which the default ``L1Manager`` does not
    provide. ``StorageManager`` narrows to this only inside the maru
    branch.
    """

    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout to the underlying maru allocator."""
        ...
