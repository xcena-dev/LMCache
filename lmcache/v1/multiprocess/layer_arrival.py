# SPDX-License-Identifier: Apache-2.0
"""Release and per-layer wait policy for layer-major KV retrieval.

Layer-major staging (see ``_run_object_group_transfer_plan_layer_major``) makes
a request's KV land one layer slice at a time. Two decisions follow from that,
and this module owns both so they can be tested without a GPU or a live server:

1. **When to hand the request back to the engine.** Reporting it only once the
   whole transfer has finished throws the pipeline away: by the time the forward
   pass starts every layer is already resident, so the per-layer wait below
   never blocks on anything and prefill runs strictly after the transfer. The
   request has to be released as soon as the first slice lands, so its own
   attention runs while the remaining slices are still moving.

   Measurements on an in-process CXL connector with the same structure put the
   first layer's arrival at 2.7-3.1 ms against 89 ms for a full transfer, at
   every concurrency from 1 to 8 -- the release condition is satisfied long
   before the transfer ends, and waiting for more only delays the start.

2. **How long each layer waits.** Layer L must not be read before its slice has
   landed, so ``event_for_layer`` hands back the event to gate on, and the
   caller (``wait_for_layer_load``) waits on it. Slices already waited on are
   reported once, so a slice covering N layers costs one wait, not N.

The transport is injected: whoever receives per-slice completion signals calls
``record_slice``, and the engine thread calls ``release_ready`` /
``event_for_layer``. Both sides touch the same state, so every method is
lock-guarded.
"""

# Standard
from typing import Any
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# A parked request is released once this many of its layers have landed. One is
# the measured optimum for the structure described above: the transfer outruns
# compute at full bandwidth, so waiting for a second slice only postpones the
# start without removing any later stall.
DEFAULT_RELEASE_AFTER_LAYERS = 1


class LayerArrivalGate:
    """Tracks which layer slices of one request have landed on the GPU.

    Args:
        num_layers: Layers the request's KV covers.
        release_after_layers: Layers that must land before the request may go
            back to the engine. Clamped to ``[1, num_layers]``.

    Raises:
        ValueError: If ``num_layers`` is not positive.
    """

    def __init__(
        self,
        num_layers: int,
        release_after_layers: int = DEFAULT_RELEASE_AFTER_LAYERS,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self._num_layers = num_layers
        self._release_after_layers = max(1, min(release_after_layers, num_layers))
        self._lock = threading.Lock()
        # layer index -> event to gate that layer's read on
        self._events: dict[int, Any] = {}
        self._layers_arrived = 0
        self._waited: set[int] = set()

    @property
    def num_layers(self) -> int:
        """Layers this request's KV covers."""
        return self._num_layers

    @property
    def release_after_layers(self) -> int:
        """Layers that must land before the request is released."""
        return self._release_after_layers

    def record_slice(self, layer_start: int, layer_count: int, event: Any) -> None:
        """Register that a slice's copies have been issued and gated by ``event``.

        Called by the transport, in the order slices were planned.

        Args:
            layer_start: First layer in the slice.
            layer_count: Layers in the slice.
            event: The object ``event_for_layer`` should return for these layers.

        Raises:
            ValueError: If the slice is empty or falls outside ``num_layers``.
        """
        if layer_count < 1:
            raise ValueError(f"layer_count must be >= 1, got {layer_count}")
        if layer_start < 0 or layer_start + layer_count > self._num_layers:
            raise ValueError(
                f"slice [{layer_start}, {layer_start + layer_count}) is outside "
                f"the request's {self._num_layers} layers"
            )
        with self._lock:
            for layer in range(layer_start, layer_start + layer_count):
                if layer not in self._events:
                    self._layers_arrived += 1
                self._events[layer] = event

    def release_ready(self) -> bool:
        """Whether enough layers have landed to hand the request back."""
        with self._lock:
            return self._layers_arrived >= self._release_after_layers

    def all_arrived(self) -> bool:
        """Whether every layer's slice has been recorded."""
        with self._lock:
            return self._layers_arrived >= self._num_layers

    def layers_arrived(self) -> int:
        """How many layers have landed so far."""
        with self._lock:
            return self._layers_arrived

    def event_for_layer(self, layer_idx: int) -> Any | None:
        """Return the event gating ``layer_idx``, or None if nothing to wait on.

        None means either that the layer's slice has not been recorded yet (the
        caller must poll the transport first) or that this slice was already
        waited on, in which case waiting again would be redundant.

        Args:
            layer_idx: Layer about to be read.

        Returns:
            The event to wait on, or None.
        """
        with self._lock:
            event = self._events.get(layer_idx)
            if event is None:
                return None
            key = id(event)
            if key in self._waited:
                return None
            self._waited.add(key)
            return event

    def reset(self) -> None:
        """Drop all recorded slices, for reuse across forward passes."""
        with self._lock:
            self._events.clear()
            self._layers_arrived = 0
            self._waited.clear()
