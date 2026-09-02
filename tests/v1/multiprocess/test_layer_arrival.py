# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.layer_arrival import (
    DEFAULT_RELEASE_AFTER_LAYERS,
    LayerArrivalGate,
    resolve_layers_per_stage,
)


class FakeEvent:
    """Stand-in for a CUDA/IPC event; identity is all the gate uses."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_release_waits_for_the_first_slice_only():
    gate = LayerArrivalGate(num_layers=32)
    assert not gate.release_ready()
    gate.record_slice(0, 1, FakeEvent("l0"))
    assert gate.release_ready()
    assert not gate.all_arrived()


def test_default_release_is_one_layer():
    assert DEFAULT_RELEASE_AFTER_LAYERS == 1
    assert LayerArrivalGate(num_layers=32).release_after_layers == 1


def test_release_threshold_is_clamped_into_range():
    assert LayerArrivalGate(4, release_after_layers=0).release_after_layers == 1
    assert LayerArrivalGate(4, release_after_layers=99).release_after_layers == 4


def test_all_arrived_only_after_every_slice():
    gate = LayerArrivalGate(num_layers=8)
    for start in range(0, 8, 4):
        assert not gate.all_arrived()
        gate.record_slice(start, 4, FakeEvent(f"s{start}"))
    assert gate.all_arrived()
    assert gate.layers_arrived() == 8


def test_a_slice_costs_one_wait_not_one_per_layer():
    gate = LayerArrivalGate(num_layers=8)
    event = FakeEvent("s0")
    gate.record_slice(0, 4, event)
    assert gate.event_for_layer(0) is event
    # Layers 1..3 share the slice, so there is nothing left to wait on.
    assert [gate.event_for_layer(i) for i in (1, 2, 3)] == [None, None, None]


def test_each_slice_is_waited_on_once():
    gate = LayerArrivalGate(num_layers=4)
    first, second = FakeEvent("a"), FakeEvent("b")
    gate.record_slice(0, 2, first)
    gate.record_slice(2, 2, second)
    assert gate.event_for_layer(0) is first
    assert gate.event_for_layer(2) is second
    assert gate.event_for_layer(1) is None
    assert gate.event_for_layer(3) is None


def test_unrecorded_layer_has_nothing_to_wait_on_yet():
    gate = LayerArrivalGate(num_layers=4)
    gate.record_slice(0, 1, FakeEvent("l0"))
    assert gate.event_for_layer(3) is None


def test_reset_clears_slices_for_reuse():
    gate = LayerArrivalGate(num_layers=4)
    gate.record_slice(0, 4, FakeEvent("s"))
    gate.reset()
    assert gate.layers_arrived() == 0
    assert not gate.release_ready()
    assert gate.event_for_layer(0) is None


@pytest.mark.parametrize(
    "layer_start,layer_count",
    [(-1, 1), (0, 0), (3, 2), (4, 1)],
)
def test_slices_outside_the_request_are_rejected(layer_start, layer_count):
    gate = LayerArrivalGate(num_layers=4)
    with pytest.raises(ValueError):
        gate.record_slice(layer_start, layer_count, FakeEvent("x"))


def test_num_layers_must_be_positive():
    with pytest.raises(ValueError):
        LayerArrivalGate(num_layers=0)


def test_re_recording_a_slice_does_not_double_count():
    gate = LayerArrivalGate(num_layers=4)
    gate.record_slice(0, 2, FakeEvent("a"))
    gate.record_slice(0, 2, FakeEvent("a-again"))
    assert gate.layers_arrived() == 2


def test_transport_and_engine_threads_can_race():
    """Recording from one thread while another polls must not lose slices."""
    gate = LayerArrivalGate(num_layers=64)
    done = threading.Event()

    def recorder():
        for start in range(64):
            gate.record_slice(start, 1, FakeEvent(f"l{start}"))
        done.set()

    def poller():
        while not done.is_set():
            gate.release_ready()
            gate.layers_arrived()

    threads = [threading.Thread(target=recorder), threading.Thread(target=poller)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert gate.layers_arrived() == 64
    assert gate.all_arrived()


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0),
        (False, 0),
        (0, 0),
        (True, 1),
        (1, 1),
        (4, 4),
        (-3, 0),
        ("0", 0),
        ("4", 4),
        ("true", 1),
        ("True", 1),
        ("on", 1),
        ("yes", 1),
        ("false", 0),
        ("off", 0),
        ("", 0),
        ("  8  ", 8),
        ("banana", 0),
        (2.5, 0),
        ([], 0),
    ],
)
def test_the_single_setting_accepts_what_a_deployer_would_write(value, expected):
    assert resolve_layers_per_stage(value) == expected
