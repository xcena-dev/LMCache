# SPDX-License-Identifier: Apache-2.0
"""Batch completion must cover the stream that actually writes the KV bytes."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer


def test_batch_copies_and_completion_share_the_registered_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambient default stream must not receive this instance's batch copies."""
    ambient = {"device": "default-device", "stream": "default-stream"}
    copy_contexts: list[tuple[str, str]] = []
    completion_streams: list[str] = []

    @contextmanager
    def select(name: str, value: str) -> Iterator[None]:
        previous = ambient[name]
        ambient[name] = value
        try:
            yield
        finally:
            ambient[name] = previous

    @contextmanager
    def read_objects(keys: list[Any]) -> Iterator[list[object]]:
        yield [SimpleNamespace(get_size=lambda: 16) for _ in keys]

    def enqueue(*args: Any, **kwargs: Any) -> None:
        copy_contexts.append((ambient["device"], ambient["stream"]))

    storage = SimpleNamespace(
        read_prefetched_results=read_objects,
        finish_write=lambda keys: None,
        finish_read_prefetched=lambda keys: None,
    )
    context = SimpleNamespace(
        storage_manager=storage,
        resolve_obj_keys=lambda key, groups: [[key.request_id]],
        chunk_size=256,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
        ),
    )
    cache = SimpleNamespace(
        device="registered-device",
        stream="registered-stream",
        cupy_stream="registered-stream",
        max_batch_size=1,
        calculate_num_blocks=lambda tokens, group: 1,
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            num_kernel_groups=1,
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1], group_kinds=["full"]
            ),
        ),
    )
    events = SimpleNamespace(
        create_event=lambda device: object(),
        import_event=lambda handle, device: object(),
        wait_event=lambda event, stream: None,
        record_event=lambda event, stream: completion_streams.append(stream),
        export_event=lambda event, device: b"completion",
    )
    monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", MagicMock)
    monkeypatch.setattr(
        transfer,
        "torch_dev",
        SimpleNamespace(
            device=lambda value: select("device", value),
            stream=lambda value: select("stream", value),
        ),
    )
    monkeypatch.setattr(transfer, "transfer_kv_per_object_group", enqueue)
    monkeypatch.setattr(
        transfer, "downsample_and_stage_block_ids", lambda context, ids: ids
    )
    monkeypatch.setattr(transfer, "submit_callback_to_stream", lambda *args: None)
    module = transfer.LMCacheDrivenTransferModule(context)
    entry = SimpleNamespace(
        cache_context=cache, event_backend=events, model_name="model"
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda instance: entry)
    try:
        _, hits = module.retrieve_batch(
            [
                SimpleNamespace(request_id="a", cache_salt=""),
                SimpleNamespace(request_id="b", cache_salt=""),
            ],
            instance_id=1,
            gpu_block_ids=[[[1]], [[2]]],
            event_ipc_handle=b"producer",
        )
    finally:
        module.close()

    assert hits == [True, True]
    assert copy_contexts == [(cache.device, cache.stream)] * 2
    assert completion_streams == [cache.stream]
    assert ambient == {"device": "default-device", "stream": "default-stream"}
