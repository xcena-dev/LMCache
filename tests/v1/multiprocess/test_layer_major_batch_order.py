# SPDX-License-Identifier: Apache-2.0
"""The order a batch's requests and slices are enqueued in, and when arrival is
published.

A batched layer-major retrieve shares one FIFO stream with every other
retrieve the worker has in flight. Two orderings have to hold together for
several requests to pay off:

* requests outer, slices inner -- every layer of request A is moved before
  any layer of request B, so A's data is complete, and A can be computing,
  while B's is still on its way. Moving layer 0 of every request first would
  leave every request waiting on the batch's last slice;

* arrival published per request, right after each of that request's slices --
  so a request leaves the wait for remote KV on its own bytes instead of the
  whole batch's.

Both are checked here by recording the calls, with no GPU involved. The block
ids are staged once per request: nothing else touches the shared staging
buffer between one request's slices.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import object_group_transfer as transfer

NUM_LAYERS = 8
LAYERS_PER_STAGE = 2
NUM_REQUESTS = 3


@pytest.fixture
def call_log(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record stage, enqueue and publish calls in the order they are made."""
    log: list[tuple] = []

    def fake_stage(cache_context: object, block_ids: list[list[int]]) -> object:
        log.append(("stage", block_ids[0][0]))
        return block_ids

    def fake_enqueue(
        cache_context: object,
        block_ids_gpu: object,
        memory_objs: list,
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: object,
        layer_start: int,
        layer_count: int,
    ) -> None:
        log.append(("enqueue", memory_objs[0], layer_start))

    monkeypatch.setattr(transfer, "downsample_and_stage_block_ids", fake_stage)
    monkeypatch.setattr(transfer, "_enqueue_object_group_layer_slice", fake_enqueue)
    monkeypatch.setattr(
        transfer, "_object_group_num_layers", lambda ctx, gid: NUM_LAYERS
    )
    return log


def _per_request() -> list[tuple[list[list[int]], list[str]]]:
    return [([[i]], [f"req{i}"]) for i in range(NUM_REQUESTS)]


def test_request_outer_with_arrival_published_per_slice(call_log: list[tuple]) -> None:
    """A request's slices are enqueued back to back, each published as it goes."""
    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=_per_request(),
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=None,
        layers_per_stage=LAYERS_PER_STAGE,
        on_request_slice=lambda position, layer_start, layer_count: call_log.append(
            ("publish", position, layer_start, layer_count)
        ),
    )

    expected: list[tuple] = []
    for position in range(NUM_REQUESTS):
        expected.append(("stage", position))
        for layer_start in range(0, NUM_LAYERS, LAYERS_PER_STAGE):
            expected.append(("enqueue", f"req{position}", layer_start))
            expected.append(("publish", position, layer_start, LAYERS_PER_STAGE))
    assert call_log == expected


def test_block_ids_are_staged_once_per_request(call_log: list[tuple]) -> None:
    """The shared staging buffer is filled once per request, not once per slice."""
    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=_per_request(),
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=None,
        layers_per_stage=LAYERS_PER_STAGE,
    )

    stages = [entry for entry in call_log if entry[0] == "stage"]
    assert stages == [("stage", i) for i in range(NUM_REQUESTS)]


def test_a_partial_last_slice_is_published_with_its_real_width(
    call_log: list[tuple],
) -> None:
    """A layer count that does not divide evenly ends with a narrower slice."""
    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=_per_request()[:1],
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=None,
        layers_per_stage=3,
        on_request_slice=lambda position, layer_start, layer_count: call_log.append(
            ("publish", position, layer_start, layer_count)
        ),
    )

    publishes = [entry for entry in call_log if entry[0] == "publish"]
    assert publishes == [
        ("publish", 0, 0, 3),
        ("publish", 0, 3, 3),
        ("publish", 0, 6, 2),
    ]


def test_no_publisher_still_moves_every_slice(call_log: list[tuple]) -> None:
    """Without a publisher the transfer runs, just with no progress reported."""
    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=_per_request(),
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=None,
        layers_per_stage=LAYERS_PER_STAGE,
    )

    slices = NUM_LAYERS // LAYERS_PER_STAGE
    enqueues = [entry for entry in call_log if entry[0] == "enqueue"]
    assert len(enqueues) == slices * NUM_REQUESTS


def test_a_slice_width_below_one_is_rejected(call_log: list[tuple]) -> None:
    """A width of zero would mean no slices at all, so it is an error."""
    with pytest.raises(ValueError, match="layers_per_stage"):
        transfer.transfer_kv_batch_layer_major(
            cache_context=object(),
            per_request=[([[0]], ["req0"])],
            object_group_id=0,
            batch_size=1,
            skip_first_n_tokens=0,
            direction=None,
            layers_per_stage=0,
        )
    assert call_log == []
