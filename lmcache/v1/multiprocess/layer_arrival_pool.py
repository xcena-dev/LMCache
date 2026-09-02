# SPDX-License-Identifier: Apache-2.0
"""Worker-side slots that track layer-slice arrival for in-flight retrieves.

Under layer-major retrieval the worker needs two things per request: a place
for the server to publish how far it has got, and one event per layer to wait
on. Both are per-request, but neither is cheap to build per request -- creating
and exporting an event costs a driver round trip -- so they are pooled.

A slot owns one board entry plus ``num_layers`` events, created once and reused.
Acquiring a slot for a retrieve resets its board entry to zero and hands back
the already-exported event handles; the server re-records the same events for
each retrieve that lands on that slot. A retrieve holds its slot until the
worker is done reading its layers.

The pool is deliberately finite. When every slot is busy, ``acquire`` returns
None and the caller submits an ordinary retrieve: the request then waits for the
whole transfer as before, losing the overlap but nothing else.
"""

# Standard
from typing import Any
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.layer_arrival import LayerArrivalGate
from lmcache.v1.multiprocess.layer_arrival_board import LayerArrivalBoard

logger = init_logger(__name__)


class LayerArrivalPool:
    """Fixed set of reusable arrival slots for one worker process.

    Args:
        num_layers: Layers a request's KV covers.
        num_slots: Concurrent retrieves that can track arrival.
        board_name: POSIX shared-memory segment name to create.
        event_backend: Device event backend providing ``create_event`` and
            ``export_event``.
        device: Device the events belong to.
        release_after_layers: Layers that must land before a request may go back
            to the engine.

    Raises:
        ValueError: If ``num_layers`` or ``num_slots`` is not positive.
        OSError: If the shared-memory segment cannot be created.
    """

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        board_name: str,
        event_backend: Any,
        device: Any,
        release_after_layers: int = 1,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self._num_layers = num_layers
        self._num_slots = num_slots
        self._release_after_layers = release_after_layers
        self._event_backend = event_backend
        self._device = device
        self._board = LayerArrivalBoard.create(board_name, num_slots)
        self._lock = threading.Lock()
        self._free: list[int] = list(range(num_slots))
        # slot -> per-layer events and their exported handles, built once.
        self._slot_events: dict[int, list[Any]] = {}
        self._slot_handles: dict[int, list[bytes]] = {}
        # request id -> (slot, gate)
        self._active: dict[str, tuple[int, LayerArrivalGate]] = {}

    @property
    def num_layers(self) -> int:
        """Layers a request's KV covers."""
        return self._num_layers

    @property
    def board_name(self) -> str:
        """Name of the shared-memory segment the server publishes into."""
        return self._board.name

    def acquire(
        self, request_id: str
    ) -> tuple[tuple[str, int, int], list[bytes]] | None:
        """Take a slot for ``request_id``.

        Args:
            request_id: The retrieve's request id.

        Returns:
            ``((segment name, slot, slots), per-layer event handles)`` to send to
            the server, or None when no slot is free or the events could not be
            built -- in which case the caller submits an ordinary retrieve.
        """
        with self._lock:
            if request_id in self._active:
                logger.warning(
                    "Request %s already holds an arrival slot; reusing it",
                    request_id,
                )
                slot = self._active[request_id][0]
            elif self._free:
                slot = self._free.pop()
            else:
                return None

            try:
                handles = self._ensure_slot_events(slot)
            except Exception:
                logger.exception(
                    "Cannot build arrival events for slot %d; this retrieve "
                    "runs without per-layer progress",
                    slot,
                )
                if slot not in [s for s, _ in self._active.values()]:
                    self._free.append(slot)
                return None

            self._board.reset(slot)
            self._active[request_id] = (
                slot,
                LayerArrivalGate(
                    self._num_layers,
                    release_after_layers=self._release_after_layers,
                ),
            )
            return (self._board.name, slot, self._num_slots), handles

    def layers_arrived(self, request_id: str) -> int:
        """Return how many of this request's layers the server has recorded.

        Reads the board and feeds what it finds into the request's gate, so a
        later :meth:`event_for_layer` knows which events are safe to wait on.

        Args:
            request_id: The retrieve's request id.

        Returns:
            The layer count, 0 if the request holds no slot.
        """
        with self._lock:
            entry = self._active.get(request_id)
            if entry is None:
                return 0
            slot, gate = entry
            arrived = min(self._board.read(slot), self._num_layers)
            events = self._slot_events.get(slot)
        if arrived and events is not None:
            for layer in range(arrived):
                gate.record_slice(layer, 1, events[layer])
        return arrived

    def release_ready(self, request_id: str) -> bool:
        """Whether enough layers have landed to hand the request to the engine.

        Args:
            request_id: The retrieve's request id.

        Returns:
            True once the release threshold is met.
        """
        with self._lock:
            entry = self._active.get(request_id)
        if entry is None:
            return False
        self.layers_arrived(request_id)
        return entry[1].release_ready()

    def event_for_layer(self, request_id: str, layer_idx: int) -> Any | None:
        """Return the event gating ``layer_idx`` for this request.

        Args:
            request_id: The retrieve's request id.
            layer_idx: Layer about to be read.

        Returns:
            The event to wait on, or None when there is nothing to wait on --
            either the layer has not been recorded yet or its slice was already
            waited on.
        """
        with self._lock:
            entry = self._active.get(request_id)
        if entry is None:
            return None
        return entry[1].event_for_layer(layer_idx)

    def active_request_ids(self) -> list[str]:
        """Return the requests currently holding a slot."""
        with self._lock:
            return list(self._active)

    def release(self, request_id: str) -> None:
        """Give up this request's slot, keeping its events for the next user.

        Args:
            request_id: The retrieve's request id.
        """
        with self._lock:
            entry = self._active.pop(request_id, None)
            if entry is None:
                return
            self._free.append(entry[0])

    def close(self) -> None:
        """Drop every slot and unlink the shared-memory segment."""
        with self._lock:
            self._active.clear()
            self._free.clear()
            self._slot_events.clear()
            self._slot_handles.clear()
            self._board.close()

    def _ensure_slot_events(self, slot: int) -> list[bytes]:
        """Build this slot's per-layer events on first use and return their handles.

        Args:
            slot: Slot index.

        Returns:
            One exported event handle per layer.
        """
        handles = self._slot_handles.get(slot)
        if handles is not None:
            return handles
        events = [
            self._event_backend.create_event(self._device)
            for _ in range(self._num_layers)
        ]
        handles = [
            self._event_backend.export_event(event, self._device) for event in events
        ]
        self._slot_events[slot] = events
        self._slot_handles[slot] = handles
        return handles
