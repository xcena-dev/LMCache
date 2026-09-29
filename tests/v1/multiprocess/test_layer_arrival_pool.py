# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.layer_arrival_board import LayerArrivalBoard
from lmcache.v1.multiprocess.layer_arrival_pool import LayerArrivalPool

NUM_LAYERS = 8
NUM_SLOTS = 2


class FakeEventBackend:
    """Hands out identity-only events and counts driver-ish calls.

    An event reports complete once the test marks it so with ``complete``;
    until then it behaves like a recorded event whose copy is still running.
    """

    def __init__(self) -> None:
        self.created = 0
        self.exported = 0
        self._done: set[int] = set()

    def create_event(self, device):
        self.created += 1
        return object()

    def export_event(self, event, device):
        self.exported += 1
        return f"handle-{id(event)}".encode()

    def query_event(self, event) -> bool:
        return id(event) in self._done

    def complete(self, event) -> None:
        self._done.add(id(event))


@pytest.fixture
def pool():
    backend = FakeEventBackend()
    p = LayerArrivalPool(
        num_layers=NUM_LAYERS,
        num_slots=NUM_SLOTS,
        board_name=f"lmc_test_pool_{os.getpid()}",
        event_backend=backend,
        device="cuda:0",
    )
    p.backend = backend  # type: ignore[attr-defined]
    try:
        yield p
    finally:
        p.close()


def _server_publishes(pool_, request_id, layers):
    """Stand in for the server: raise the board count for the request's slot."""
    board = LayerArrivalBoard.attach(pool_.board_name, NUM_SLOTS)
    try:
        slot = pool_._active[request_id][0]  # noqa: SLF001 - test reaches in
        board.publish(slot, layers)
    finally:
        board.close()


def test_acquire_hands_out_the_board_and_one_handle_per_layer(pool):
    acquired = pool.acquire("r1")
    assert acquired is not None
    (name, slot, slots), handles = acquired
    assert name == pool.board_name
    assert slots == NUM_SLOTS
    assert 0 <= slot < NUM_SLOTS
    assert len(handles) == NUM_LAYERS


def test_nothing_has_arrived_before_the_server_publishes(pool):
    pool.acquire("r1")
    assert pool.layers_arrived("r1") == 0
    assert not pool.release_ready("r1")
    assert pool.event_for_layer("r1", 0) is None


def test_release_waits_for_the_first_layer_to_land_not_just_be_recorded(pool):
    """The board says the server enqueued the copy; the event says it finished.

    Release must read both: a board count alone is raised at enqueue time,
    which with several retrieves queued on one stream can be long before the
    bytes are on the GPU.
    """
    pool.acquire("r1")
    _server_publishes(pool, "r1", 1)
    assert pool.layers_arrived("r1") == 1
    assert not pool.release_ready("r1")

    pool.backend.complete(pool.event_for_layer("r1", 0))
    assert pool.release_ready("r1")


def test_release_is_gated_on_the_configured_layer_when_more_than_one(pool):
    """With release_after_layers=N the Nth layer's event is the one queried."""
    gated = LayerArrivalPool(
        num_layers=NUM_LAYERS,
        num_slots=NUM_SLOTS,
        board_name=f"lmc_test_pool_gated_{os.getpid()}",
        event_backend=pool.backend,
        device="cuda:0",
        release_after_layers=3,
    )
    try:
        gated.acquire("r1")
        _server_publishes(gated, "r1", 3)
        first = gated.event_for_layer("r1", 0)
        pool.backend.complete(first)
        assert not gated.release_ready("r1")

        pool.backend.complete(gated.event_for_layer("r1", 2))
        assert gated.release_ready("r1")
    finally:
        gated.close()


def test_a_completed_event_is_not_enough_without_the_record(pool):
    """An event that was never recorded reports complete; the board guards it."""
    pool.acquire("r1")
    # Reach the slot's first event the way the worker would after a record.
    slot = pool._active["r1"][0]  # noqa: SLF001 - test reaches in
    pool.backend.complete(pool._slot_events[slot][0])  # noqa: SLF001
    assert not pool.release_ready("r1")


def test_only_recorded_layers_are_waitable(pool):
    pool.acquire("r1")
    _server_publishes(pool, "r1", 3)
    assert pool.layers_arrived("r1") == 3
    assert pool.event_for_layer("r1", 0) is not None
    assert pool.event_for_layer("r1", 2) is not None
    # Layer 3 has not been published, so there is nothing safe to wait on.
    assert pool.event_for_layer("r1", 3) is None


def test_each_layer_is_waited_on_once(pool):
    pool.acquire("r1")
    _server_publishes(pool, "r1", NUM_LAYERS)
    pool.layers_arrived("r1")
    first = [pool.event_for_layer("r1", i) for i in range(NUM_LAYERS)]
    again = [pool.event_for_layer("r1", i) for i in range(NUM_LAYERS)]
    assert all(e is not None for e in first)
    assert again == [None] * NUM_LAYERS


def test_a_count_past_the_layer_count_is_clamped(pool):
    pool.acquire("r1")
    _server_publishes(pool, "r1", NUM_LAYERS + 5)
    assert pool.layers_arrived("r1") == NUM_LAYERS


def test_slots_run_out_and_the_caller_is_told(pool):
    assert pool.acquire("r1") is not None
    assert pool.acquire("r2") is not None
    assert pool.acquire("r3") is None


def test_releasing_frees_the_slot_for_reuse(pool):
    pool.acquire("r1")
    pool.acquire("r2")
    assert pool.acquire("r3") is None
    pool.release("r1")
    assert pool.acquire("r3") is not None


def test_events_are_built_once_per_slot_not_per_retrieve(pool):
    pool.acquire("r1")
    after_first = pool.backend.created
    assert after_first == NUM_LAYERS
    pool.release("r1")
    pool.acquire("r2")
    # r2 lands on the freed slot and reuses its events.
    assert pool.backend.created == after_first


def test_a_reused_slot_starts_from_zero(pool):
    pool.acquire("r1")
    _server_publishes(pool, "r1", NUM_LAYERS)
    assert pool.layers_arrived("r1") == NUM_LAYERS
    pool.release("r1")
    pool.acquire("r2")
    assert pool.layers_arrived("r2") == 0


def test_unknown_requests_report_nothing(pool):
    assert pool.layers_arrived("nope") == 0
    assert not pool.release_ready("nope")
    assert pool.event_for_layer("nope", 0) is None
    pool.release("nope")


def test_active_requests_are_listed(pool):
    pool.acquire("r1")
    pool.acquire("r2")
    assert sorted(pool.active_request_ids()) == ["r1", "r2"]
    pool.release("r1")
    assert pool.active_request_ids() == ["r2"]


@pytest.mark.parametrize("layers,slots", [(0, 1), (1, 0), (-1, 1)])
def test_bad_geometry_is_rejected(layers, slots):
    with pytest.raises(ValueError):
        LayerArrivalPool(
            num_layers=layers,
            num_slots=slots,
            board_name=f"lmc_test_bad_{os.getpid()}_{layers}_{slots}",
            event_backend=FakeEventBackend(),
            device="cuda:0",
        )
