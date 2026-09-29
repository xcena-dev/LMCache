# SPDX-License-Identifier: Apache-2.0
"""The order a batch's slices are enqueued in, and when arrival is published.

Two orderings have to hold together for a batched layer-major retrieve to pay
off with several requests in flight:

* slices outer, requests inner -- the forward pass reads layer L of every
  request before layer L+1 of any, so the batch's layer 0 must be complete
  before layer 1 starts;
* arrival published per request, right after that request's slice -- so a
  request leaves the wait for remote KV on its own bytes instead of the whole
  batch's.

The second is what an earlier version got wrong: it published once per slice
for the whole batch, so no request was released until every request's slice had
landed. Both are checked here by recording the calls, with no GPU involved.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer

NUM_LAYERS = 8
LAYERS_PER_STAGE = 2
NUM_REQUESTS = 3


@pytest.fixture
def call_log(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record enqueue and publish calls in the order they are made."""
    log: list[tuple] = []

    def fake_stage(cache_context: object, block_ids: list[list[int]]) -> object:
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


def test_slice_outer_with_arrival_published_per_request(call_log: list[tuple]) -> None:
    """Each request hears about its slice before the next request is enqueued."""
    per_request = [([[0]], [f"req{i}"]) for i in range(NUM_REQUESTS)]

    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=per_request,
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
    for layer_start in range(0, NUM_LAYERS, LAYERS_PER_STAGE):
        for position in range(NUM_REQUESTS):
            expected.append(("enqueue", f"req{position}", layer_start))
            expected.append(("publish", position, layer_start, LAYERS_PER_STAGE))
    assert call_log == expected


def test_no_publisher_still_moves_every_slice(call_log: list[tuple]) -> None:
    """Without a publisher the transfer runs, just with no progress reported."""
    per_request = [([[0]], [f"req{i}"]) for i in range(NUM_REQUESTS)]

    transfer.transfer_kv_batch_layer_major(
        cache_context=object(),
        per_request=per_request,
        object_group_id=0,
        batch_size=1,
        skip_first_n_tokens=0,
        direction=None,
        layers_per_stage=LAYERS_PER_STAGE,
    )

    slices = NUM_LAYERS // LAYERS_PER_STAGE
    assert len(call_log) == slices * NUM_REQUESTS
    assert all(entry[0] == "enqueue" for entry in call_log)


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
