# SPDX-License-Identifier: Apache-2.0
"""Layer-major retrieve moves the same bytes as the chunk-major path, on GPU.

The layer-major path slices the layer axis and issues one scatter per slice,
addressing the engine KV cache with ``layer_offset`` and the staging buffer
with the slice-relative slot. Everything below the kernel was checked by
replaying the copy descriptors on the host; this module runs the real server,
the real CUDA IPC path and the real kernels, and compares the GPU bytes.

Each slice width retrieves into its own block range, so a width that dropped,
duplicated or misplaced a layer shows up as a mismatch against the source
blocks rather than being masked by a previous retrieve.
"""

# Standard
from typing import Any, Generator
import multiprocessing as mp
import os
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.layer_arrival_board import LayerArrivalBoard
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

SERVER_HOST = "localhost"
SERVER_PORT = 5607
SERVER_URL = f"tcp://{SERVER_HOST}:{SERVER_PORT}"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0
DEFAULT_TIMEOUT = 30.0

NUM_LAYERS = 32
NUM_PAGES = 1024
PAGE_SIZE = 16
BLOCKS_PER_KEY = CHUNK_SIZE // PAGE_SIZE
NUM_KEYS = 4
BLOCKS_PER_RANGE = BLOCKS_PER_KEY * NUM_KEYS

#: Slice widths to exercise. 0 is the chunk-major path (the reference), 3 does
#: not divide 32 so the last slice is short, and 32 is one slice for everything.
SLICE_WIDTHS = (0, 1, 2, 3, 8, 32)

pytestmark = pytest.mark.cuda


def _has_working_new_shared_cuda() -> bool:
    try:
        buf = torch.empty(1024, device=torch_device_type)
        return buf.untyped_storage()._share_cuda_() is not None
    except Exception:
        return False


if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402

if not _has_working_new_shared_cuda():
    pytest.skip("new_shared_cuda is not usable here", allow_module_level=True)


class ClientContext:
    """GPU KV cache tensors plus the IPC wrappers the server registers."""

    def __init__(self, device: torch.device) -> None:
        torch.random.manual_seed(4242)
        self.device = device
        self.num_layers = NUM_LAYERS
        self.gpu_kv_caches = [
            torch.rand(
                (2, NUM_PAGES, PAGE_SIZE, 8, 128),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(NUM_LAYERS)
        ]

    def get_kv_cache(self) -> KVCache:
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]


def create_cache_key(
    index: int, round_tag: str = "store", model: str = "testmodel"
) -> IPCCacheServerKey:
    """Build a key for chunk ``index``.

    ``round_tag`` only varies the request id, which names the lookup job; the
    token ids decide the chunk hash, so every round addresses the same stored
    bytes. A retrieve must be preceded by its own lookup -- the lookup is what
    takes the object's read lock -- so each round needs its own job name.
    """
    token_ids = [index] * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"layermajor_{round_tag}_{index}",
    )


def lookup_all(client: RequestClient, keys: list[IPCCacheServerKey]) -> int:
    total = 0
    for key in keys:
        lookup_key = key.no_worker_id_version()
        client.lookup(lookup_key, 1).result(timeout=DEFAULT_TIMEOUT)
        while True:
            result = client.query_prefetch_status(lookup_key.request_id).result(
                timeout=DEFAULT_TIMEOUT
            )
            if result is not None:
                total += result
                break
    return total


def wait_until_all_stored(
    client: RequestClient, keys: list[IPCCacheServerKey], timeout: float = 30.0
) -> int:
    """Poll the lookup until every key is visible, or the timeout runs out.

    A store that has been acknowledged is not yet guaranteed to be visible to
    the next lookup, so a single lookup can under-report. Polling turns that
    into a readiness gate instead of a flaky assertion.
    """
    deadline = time.time() + timeout
    found = 0
    while time.time() < deadline:
        found = lookup_all(client, keys)
        if found == len(keys):
            return found
        time.sleep(0.2)
    return found


def store_keys(
    client: RequestClient,
    keys: list[IPCCacheServerKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: Any,
) -> None:
    for i, key in enumerate(keys):
        block_ids = gpu_block_ids[i * BLOCKS_PER_KEY : (i + 1) * BLOCKS_PER_KEY]
        result = (
            client.store(key, instance_id, [block_ids], event.ipc_handle())
            .to_device_future()
            .result(timeout=DEFAULT_TIMEOUT)
        )
        assert result is True, f"store failed for key {i}"


def retrieve_keys(
    client: RequestClient,
    keys: list[IPCCacheServerKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: Any,
    layers_per_stage: int,
    board: "LayerArrivalBoard | None" = None,
    slot: int = 0,
    layer_events: "list[Any] | None" = None,
) -> tuple[list[bool], list[int]]:
    """Retrieve every key, optionally layer-major.

    Returns:
        ``(per-key success, per-key board count after the retrieve)``. The board
        count is 0 when the retrieve did not ask for per-slice progress.
    """
    results: list[bool] = []
    published: list[int] = []
    for i, key in enumerate(keys):
        block_ids = gpu_block_ids[i * BLOCKS_PER_KEY : (i + 1) * BLOCKS_PER_KEY]
        arrival_board: tuple[str, int, int] | None = None
        layer_event_handles: list[bytes] | None = None
        if layers_per_stage > 0:
            assert board is not None and layer_events is not None
            board.reset(slot)
            arrival_board = (board.name, slot, board.num_slots)
            layer_event_handles = [e.ipc_handle() for e in layer_events]
        result = (
            client.retrieve(
                key,
                instance_id,
                [block_ids],
                event.ipc_handle(),
                0,
                arrival_board,
                layer_event_handles,
                layers_per_stage if arrival_board is not None else 0,
            )
            .to_device_future()
            .result(timeout=DEFAULT_TIMEOUT)
        )
        results.append(result)
        published.append(board.read(slot) if layers_per_stage > 0 and board else 0)
    return results, published


def server_process_runner(host: str, port: int) -> None:
    run_cache_server(
        mp_config=MPServerConfig(host=host, port=port, chunk_size=CHUNK_SIZE),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=int(CPU_BUFFER_SIZE * 1024**3),
                    use_lazy=True,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(scope="module")
def server_process() -> Generator[mp.Process, None, None]:
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner, args=(SERVER_HOST, SERVER_PORT), daemon=True
    )
    process.start()
    time.sleep(2)
    yield process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="module")
def zmq_context() -> Generator[zmq.Context, None, None]:
    yield zmq.Context.instance()


@pytest.fixture(scope="module")
def client(
    server_process: mp.Process, zmq_context: zmq.Context
) -> Generator[RequestClient, None, None]:
    request_client = RequestClientFactory.create(SERVER_URL, context=zmq_context)
    yield request_client
    request_client.close()


@pytest.fixture(scope="module")
def client_context() -> Generator[ClientContext, None, None]:
    ctx = ClientContext(device=torch.device(torch_device_type))
    yield ctx
    del ctx.gpu_kv_caches
    torch_dev.empty_cache()


@pytest.fixture(scope="module")
def registered_instance(
    client: RequestClient, client_context: ClientContext
) -> Generator[int, None, None]:
    instance_id = os.getpid()
    client.register_kv_cache(
        instance_id,
        client_context.get_kv_cache(),
        "testmodel",
        1,
        EngineType.VLLM,
        {},
        [],
    ).result(timeout=DEFAULT_TIMEOUT)
    yield instance_id
    try:
        client.clear().result(timeout=DEFAULT_TIMEOUT)
        client.unregister_kv_cache(instance_id).result(timeout=DEFAULT_TIMEOUT)
    except Exception as exc:  # pragma: no cover - teardown best effort
        print(f"unregister failed: {exc}")


def test_layer_major_retrieve_matches_the_source_blocks(
    client: RequestClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """Every slice width reproduces the stored blocks, layer for layer.

    The source blocks are stored once, then each width retrieves into its own
    destination range. A width that dropped a layer, wrote it twice or put it at
    the wrong offset differs from the source there.
    """
    store_keys_ = [create_cache_key(i, "a") for i in range(NUM_KEYS)]
    source_blocks = list(range(0, BLOCKS_PER_RANGE))

    event = torch_dev.Event(interprocess=True)
    event.record()
    store_keys(client, store_keys_, registered_instance, source_blocks, event)
    assert wait_until_all_stored(client, store_keys_) == NUM_KEYS

    source = [
        client_context.gpu_kv_caches[layer][:, :BLOCKS_PER_RANGE].clone()
        for layer in range(NUM_LAYERS)
    ]

    board = LayerArrivalBoard.create(f"lmcache_test_arrival_{os.getpid()}", 2)
    layer_events = [torch_dev.Event(interprocess=True) for _ in range(NUM_LAYERS)]
    for layer_event in layer_events:
        layer_event.record()

    try:
        for index, width in enumerate(SLICE_WIDTHS):
            offset = BLOCKS_PER_RANGE * (index + 1)
            dest_blocks = list(range(offset, offset + BLOCKS_PER_RANGE))
            event = torch_dev.Event(interprocess=True)
            event.record()

            # A retrieve reads what its own lookup unlocked, so each round
            # looks the chunks up again under a fresh job name.
            round_keys = [
                create_cache_key(i, f"w{width}_{index}") for i in range(NUM_KEYS)
            ]
            assert lookup_all(client, round_keys) == NUM_KEYS, (
                f"width={width}: the stored chunks are not visible to lookup"
            )

            results, published = retrieve_keys(
                client,
                round_keys,
                registered_instance,
                dest_blocks,
                event,
                layers_per_stage=width,
                board=board,
                slot=0,
                layer_events=layer_events,
            )
            assert all(results), f"width={width}: retrieve reported a miss"
            torch_dev.synchronize()

            for layer in range(NUM_LAYERS):
                got = client_context.gpu_kv_caches[layer][
                    :, offset : offset + BLOCKS_PER_RANGE
                ]
                assert torch.equal(source[layer], got), (
                    f"width={width}, layer={layer}: retrieved bytes differ "
                    "from the stored blocks"
                )

            if width > 0:
                assert all(count == NUM_LAYERS for count in published), (
                    f"width={width}: the server published {published} slices "
                    f"instead of {NUM_LAYERS} layers for every key"
                )
    finally:
        board.close()


def test_layer_major_and_chunk_major_agree_bit_for_bit(
    client: RequestClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """One layer per slice lands exactly what the chunk-major path lands.

    Comparing the two destination ranges directly, rather than each against the
    source, is the statement the note makes: turning the knob on changes when
    bytes arrive, not which bytes.
    """
    store_keys_ = [create_cache_key(100 + i, "ab") for i in range(NUM_KEYS)]
    source_blocks = list(range(0, BLOCKS_PER_RANGE))

    event = torch_dev.Event(interprocess=True)
    event.record()
    store_keys(client, store_keys_, registered_instance, source_blocks, event)
    assert wait_until_all_stored(client, store_keys_) == NUM_KEYS

    chunk_offset = BLOCKS_PER_RANGE * 8
    layer_offset = BLOCKS_PER_RANGE * 9

    chunk_keys = [create_cache_key(100 + i, "ab_chunk") for i in range(NUM_KEYS)]
    assert lookup_all(client, chunk_keys) == NUM_KEYS
    event = torch_dev.Event(interprocess=True)
    event.record()
    results, _ = retrieve_keys(
        client,
        chunk_keys,
        registered_instance,
        list(range(chunk_offset, chunk_offset + BLOCKS_PER_RANGE)),
        event,
        layers_per_stage=0,
    )
    assert all(results)

    board = LayerArrivalBoard.create(f"lmcache_test_arrival_ab_{os.getpid()}", 2)
    layer_events = [torch_dev.Event(interprocess=True) for _ in range(NUM_LAYERS)]
    for layer_event in layer_events:
        layer_event.record()
    try:
        layer_keys = [create_cache_key(100 + i, "ab_layer") for i in range(NUM_KEYS)]
        assert lookup_all(client, layer_keys) == NUM_KEYS
        event = torch_dev.Event(interprocess=True)
        event.record()
        results, published = retrieve_keys(
            client,
            layer_keys,
            registered_instance,
            list(range(layer_offset, layer_offset + BLOCKS_PER_RANGE)),
            event,
            layers_per_stage=1,
            board=board,
            slot=1,
            layer_events=layer_events,
        )
        assert all(results)
        assert all(count == NUM_LAYERS for count in published)
    finally:
        board.close()

    torch_dev.synchronize()
    for layer in range(NUM_LAYERS):
        chunk_major = client_context.gpu_kv_caches[layer][
            :, chunk_offset : chunk_offset + BLOCKS_PER_RANGE
        ]
        layer_major = client_context.gpu_kv_caches[layer][
            :, layer_offset : layer_offset + BLOCKS_PER_RANGE
        ]
        assert torch.equal(chunk_major, layer_major), (
            f"layer={layer}: the two paths wrote different bytes"
        )


def test_batch_retrieve_moves_every_request_correctly(
    client: RequestClient,
    client_context: ClientContext,
    registered_instance: int,
):
    """A batched layer-major retrieve lands the right bytes for every request.

    The batch path walks slices outer and requests inner, so several requests
    share the staging slots within one slice. Each (slice, request) pair is its
    own native call and stream order keeps that reuse safe -- but only if the
    ordering really is what the code intends, which is what this checks: each
    request's destination range must match the blocks it stored.

    Each request also has its own board slot, and every slot must end up with
    all the layers published, or a request would be left waiting on progress
    that went to someone else's slot.
    """
    store_keys_ = [create_cache_key(200 + i, "batch") for i in range(NUM_KEYS)]
    source_blocks = list(range(0, BLOCKS_PER_RANGE))

    event = torch_dev.Event(interprocess=True)
    event.record()
    store_keys(client, store_keys_, registered_instance, source_blocks, event)
    assert wait_until_all_stored(client, store_keys_) == NUM_KEYS

    # Each key was stored from its own slice of the source range, so request i
    # must come back with slice i -- comparing them all against slice 0 would
    # pass only if the batch path collapsed the requests together.
    source = [
        client_context.gpu_kv_caches[layer][:, :BLOCKS_PER_RANGE].clone()
        for layer in range(NUM_LAYERS)
    ]

    # One slot per request, the way the worker's pool hands them out.
    board = LayerArrivalBoard.create(
        f"lmcache_test_arrival_batch_{os.getpid()}", NUM_KEYS
    )
    layer_events = [
        [torch_dev.Event(interprocess=True) for _ in range(NUM_LAYERS)]
        for _ in range(NUM_KEYS)
    ]
    for request_events in layer_events:
        for layer_event in request_events:
            layer_event.record()
    try:
        # One key per request, each retrieving the same stored chunk into its
        # own destination range, so a request that read another's staging slot
        # shows up as a mismatch.
        round_keys = [create_cache_key(200 + i, f"batch_r{i}") for i in range(NUM_KEYS)]
        assert lookup_all(client, round_keys) == NUM_KEYS

        offsets = [BLOCKS_PER_RANGE * (10 + i) for i in range(NUM_KEYS)]
        block_ids = [[list(range(off, off + BLOCKS_PER_KEY))] for off in offsets]
        for slot in range(NUM_KEYS):
            board.reset(slot)
        event = torch_dev.Event(interprocess=True)
        event.record()
        # to_device_future() consumes the event handle and hands back the second
        # element of the response, so this is the per-request hit list.
        hits = (
            client.retrieve_batch(
                round_keys,
                registered_instance,
                block_ids,
                event.ipc_handle(),
                0,
                [(board.name, s, board.num_slots) for s in range(NUM_KEYS)],
                [
                    [e.ipc_handle() for e in request_events]
                    for request_events in layer_events
                ],
                8,
            )
            .to_device_future()
            .result(timeout=DEFAULT_TIMEOUT)
        )
        assert all(hits), f"batch retrieve reported misses: {hits}"
        for slot in range(NUM_KEYS):
            assert board.read(slot) == NUM_LAYERS, (
                f"request {slot}: the server published {board.read(slot)} "
                f"layers instead of {NUM_LAYERS}"
            )
        torch_dev.synchronize()

        for i, off in enumerate(offsets):
            src_start = i * BLOCKS_PER_KEY
            for layer in range(NUM_LAYERS):
                got = client_context.gpu_kv_caches[layer][:, off : off + BLOCKS_PER_KEY]
                want = source[layer][:, src_start : src_start + BLOCKS_PER_KEY]
                assert torch.equal(want, got), (
                    f"request {i}, layer {layer}: the batch path wrote the wrong "
                    "bytes into this request's blocks"
                )
    finally:
        board.close()
