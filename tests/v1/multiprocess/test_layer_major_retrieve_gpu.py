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
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.server import run_cache_server

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
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402

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


def lookup_all(client: MessageQueueClient, keys: list[IPCCacheServerKey]) -> int:
    total = 0
    for key in keys:
        lookup_key = key.no_worker_id_version()
        client.submit_request(
            RequestType.LOOKUP,
            [lookup_key, 1],
            get_response_class(RequestType.LOOKUP),
        ).result(timeout=DEFAULT_TIMEOUT)
        while True:
            result = client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [lookup_key.request_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=DEFAULT_TIMEOUT)
            if result is not None:
                total += result
                break
    return total


def wait_until_all_stored(
    client: MessageQueueClient, keys: list[IPCCacheServerKey], timeout: float = 30.0
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
    client: MessageQueueClient,
    keys: list[IPCCacheServerKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: Any,
) -> None:
    for i, key in enumerate(keys):
        block_ids = gpu_block_ids[i * BLOCKS_PER_KEY : (i + 1) * BLOCKS_PER_KEY]
        result = (
            client.submit_request(
                RequestType.STORE,
                [key, instance_id, [block_ids], event.ipc_handle()],
                get_response_class(RequestType.STORE),
            )
            .to_device_future()
            .result(timeout=DEFAULT_TIMEOUT)
        )
        assert result is True, f"store failed for key {i}"


def retrieve_keys(
    client: MessageQueueClient,
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
        payload: list[Any] = [key, instance_id, [block_ids], event.ipc_handle(), 0]
        if layers_per_stage > 0:
            assert board is not None and layer_events is not None
            board.reset(slot)
            payload += [
                (board.name, slot, board.num_slots),
                [e.ipc_handle() for e in layer_events],
                layers_per_stage,
            ]
        else:
            payload += [None, None, 0]
        result = (
            client.submit_request(
                RequestType.RETRIEVE,
                payload,
                get_response_class(RequestType.RETRIEVE),
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
) -> Generator[MessageQueueClient, None, None]:
    mq_client = MessageQueueClient(server_url=SERVER_URL, context=zmq_context)
    yield mq_client
    mq_client.close()


@pytest.fixture(scope="module")
def client_context() -> Generator[ClientContext, None, None]:
    ctx = ClientContext(device=torch.device(torch_device_type))
    yield ctx
    del ctx.gpu_kv_caches
    torch_dev.empty_cache()


@pytest.fixture(scope="module")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    instance_id = os.getpid()
    client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [
            instance_id,
            client_context.get_kv_cache(),
            "testmodel",
            1,
            EngineType.VLLM,
            {},
            [],
        ],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    ).result(timeout=DEFAULT_TIMEOUT)
    yield instance_id
    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
    except Exception as exc:  # pragma: no cover - teardown best effort
        print(f"unregister failed: {exc}")


def test_layer_major_retrieve_matches_the_source_blocks(
    client: MessageQueueClient,
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
    client: MessageQueueClient,
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
