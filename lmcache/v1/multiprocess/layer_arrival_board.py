# SPDX-License-Identifier: Apache-2.0
"""Cross-process record of how far a layer-major retrieve has progressed.

Layer-major retrieval copies a request's KV a layer slice at a time. To start
the model on layer 0 while later slices are still moving, the worker process
must know which slices have landed -- but the copies are issued by the LMCache
server, a different process.

The obvious shortcut does not work. Handing the server a pre-made event per
slice and letting the worker poll them is unsound: an event that has not been
recorded yet reports *complete*, so the worker would read KV that was never
copied. The existing single-event path avoids this by only touching the event
after the server's response has arrived, which is exactly the wait we are
trying to remove.

So the two facts are carried separately:

* **"the server has recorded slice i"** -- this board. A monotonic count per
  in-flight retrieve, written by the server after it records the slice's event
  and read by the worker.
* **"slice i's bytes have landed on the GPU"** -- the slice's own event, which
  only becomes meaningful once the board says it was recorded.

The board is a small POSIX shared-memory segment of fixed-size slots, one slot
per in-flight retrieve. Each slot holds one 64-bit count, and each sits on its
own cache line so concurrent slots never share one. Every slot has a single
writer (the server, for the retrieve holding that slot) and a single reader
(the worker), so a plain aligned store and load are enough; no lock is taken on
either side.
"""

# Standard
import ctypes

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_map_readwrite,
    shm_munmap,
    shm_unlink,
)

logger = init_logger(__name__)

#: Bytes per slot. One cache line, so neighbouring slots never share one.
SLOT_BYTES = 64


class LayerArrivalBoard:
    """A fixed set of slots recording per-retrieve layer-slice progress.

    Construct with :meth:`create` on the side that owns the segment (the
    worker, which also owns the retrieve) and :meth:`attach` on the side that
    writes progress (the server).

    Args:
        name: POSIX shared-memory segment name.
        num_slots: Slots in the segment. Must match on both sides.
        addr: Mapped base address of the segment.
        owner: Whether this handle created the segment and should unlink it.
    """

    def __init__(self, name: str, num_slots: int, addr: int, owner: bool) -> None:
        self._name = name
        self._num_slots = num_slots
        self._addr = addr
        self._owner = owner
        self._nbytes = num_slots * SLOT_BYTES
        # One c_uint64 per 8 bytes across the whole segment, indexed with a
        # stride of a full slot, so slot i lands on its own cache line rather
        # than on the i-th consecutive 8 bytes.
        self._stride = SLOT_BYTES // ctypes.sizeof(ctypes.c_uint64)
        self._words = (ctypes.c_uint64 * (num_slots * self._stride)).from_address(addr)

    @classmethod
    def create(cls, name: str, num_slots: int) -> "LayerArrivalBoard":
        """Create the segment and zero every slot.

        Args:
            name: POSIX shared-memory segment name.
            num_slots: Slots to allocate.

        Returns:
            A board that owns the segment.

        Raises:
            ValueError: If ``num_slots`` is not positive.
            OSError: If the segment already exists or cannot be created.
        """
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        nbytes = num_slots * SLOT_BYTES
        addr = shm_create_readwrite(name, nbytes)
        board = cls(name, num_slots, addr, owner=True)
        ctypes.memset(addr, 0, nbytes)
        return board

    @classmethod
    def attach(cls, name: str, num_slots: int) -> "LayerArrivalBoard":
        """Map an existing segment without clearing it.

        Args:
            name: POSIX shared-memory segment name.
            num_slots: Slots the segment holds. Must match the creator.

        Returns:
            A board that does not own the segment.

        Raises:
            ValueError: If ``num_slots`` is not positive.
            OSError: If the segment cannot be opened or the size disagrees.
        """
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        addr = shm_map_readwrite(name, num_slots * SLOT_BYTES)
        return cls(name, num_slots, addr, owner=False)

    @property
    def name(self) -> str:
        """POSIX shared-memory segment name."""
        return self._name

    @property
    def num_slots(self) -> int:
        """Slots the board holds."""
        return self._num_slots

    def reset(self, slot: int) -> None:
        """Clear a slot before handing it to a new retrieve.

        Args:
            slot: Slot index.

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        self._words[self._checked(slot) * self._stride] = 0

    def publish(self, slot: int, slices_recorded: int) -> None:
        """Record that ``slices_recorded`` slices have had their events recorded.

        Called by the server, once per slice, with a value that only ever
        grows. The caller must have recorded the slice's event *before* calling
        this, since the count is what makes that event safe to wait on.

        Args:
            slot: Slot index for this retrieve.
            slices_recorded: Slices recorded so far, counting from 1.

        Raises:
            IndexError: If ``slot`` is out of range.
            ValueError: If ``slices_recorded`` is negative.
        """
        if slices_recorded < 0:
            raise ValueError(
                f"slices_recorded must be non-negative, got {slices_recorded}"
            )
        self._words[self._checked(slot) * self._stride] = slices_recorded

    def read(self, slot: int) -> int:
        """Return how many slices the server has recorded for this retrieve.

        Args:
            slot: Slot index for this retrieve.

        Returns:
            The count, 0 if nothing has been recorded yet.

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        return int(self._words[self._checked(slot) * self._stride])

    def close(self) -> None:
        """Unmap the segment, and unlink it if this handle created it."""
        if self._addr == 0:
            return
        addr, self._addr = self._addr, 0
        # Drop the ctypes view before unmapping: it borrows the mapping.
        del self._words
        shm_munmap(addr, self._nbytes)
        if self._owner:
            try:
                shm_unlink(self._name)
            except FileNotFoundError:
                pass

    def _checked(self, slot: int) -> int:
        """Validate a slot index.

        Args:
            slot: Slot index.

        Returns:
            The index unchanged.

        Raises:
            IndexError: If the index is out of range.
        """
        if slot < 0 or slot >= self._num_slots:
            raise IndexError(
                f"slot {slot} out of range for a {self._num_slots}-slot board"
            )
        return slot
