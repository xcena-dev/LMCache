# SPDX-License-Identifier: Apache-2.0
"""The worker really does start a request on its first layers, and safely.

The companion module ``test_layer_major_retrieve_gpu`` speaks the message
protocol directly, so it proves the server copies the right bytes but says
nothing about the half that produces the benefit: the worker handing a request
back to the engine before its last layer has landed, then blocking per layer so
the model never reads KV that is still in flight.

This module drives the real ``LMCacheMPWorkerAdapter`` against a real server, so
the arrival board, the per-layer events and the release decision are the
deployed ones. There is no vLLM here -- the engine is replaced by the loop a
worker actually runs: poll ``get_finished``, then walk the layers calling
``wait_for_layer_load`` before reading each one.
"""

# Standard
from typing import Generator
import itertools
import multiprocessing as mp
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.server import run_cache_server

SERVER_HOST = "localhost"
#: Each test gets its own server on its own port. A retrieve rides on a lookup
#: that holds the object's read lock, and this harness has no session teardown
#: to release it, so a shared server carries one test's locks into the next.
_BASE_PORT = 5641
CHUNK_SIZE = 256
NUM_LAYERS = 32
PAGE_SIZE = 16
#: Big enough that the copy is still running while the layer walk reads: at
#: this size a whole retrieve takes tens of milliseconds, so a per-layer wait
#: that did nothing would be caught reading bytes that had not landed.
NUM_PAGES = 2048
TOKENS = 8192
BLOCKS = TOKENS // PAGE_SIZE
DEFAULT_TIMEOUT = 60.0

_PORT_COUNTER = itertools.count()

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)


def parallel_strategy() -> ParallelStrategy:
    """Single-worker, single-server layout."""
    return ParallelStrategy(
        mla_only=False,
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )


def layer_name(index: int) -> str:
    """vLLM's name for a layer, which is how the adapter finds its index."""
    return f"model.layers.{index}.self_attn"


def server_runner(port: int) -> None:
    run_cache_server(
        mp_config=MPServerConfig(host=SERVER_HOST, port=port, chunk_size=CHUNK_SIZE),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=8 * 1024**3, use_lazy=True
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture
def server_url() -> Generator[str, None, None]:
    """Start a server just for this test and hand back its URL."""
    mp.set_start_method("spawn", force=True)
    port = _BASE_PORT + next(_PORT_COUNTER)
    process = mp.Process(target=server_runner, args=(port,), daemon=True)
    process.start()
    time.sleep(3)
    yield f"tcp://{SERVER_HOST}:{port}"
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="module")
def kv_caches() -> Generator[dict[str, torch.Tensor], None, None]:
    torch.random.manual_seed(11)
    caches = {
        layer_name(i): torch.rand(
            (2, NUM_PAGES, PAGE_SIZE, 8, 128),
            dtype=torch.bfloat16,
            device=torch.device(torch_device_type),
        )
        for i in range(NUM_LAYERS)
    }
    yield caches
    caches.clear()
    torch_dev.empty_cache()


def make_adapter(server_url: str, layers_per_stage: int) -> LMCacheMPWorkerAdapter:
    """Build a worker adapter talking to the real server.

    Args:
        layers_per_stage: Value for ``lmcache.mp.layerwise_overlap``. 0 keeps
            the chunk-major path.

    Returns:
        The adapter. The caller registers the KV caches.
    """
    return LMCacheMPWorkerAdapter(
        server_url=server_url,
        context=zmq.Context.instance(),
        model_name="testmodel",
        vllm_block_size=PAGE_SIZE,
        parallel_strategy=parallel_strategy(),
        mq_timeout=DEFAULT_TIMEOUT,
        extra_config={"lmcache.mp.layerwise_overlap": layers_per_stage},
    )


def make_scheduler(server_url: str) -> LMCacheMPSchedulerAdapter:
    """Build the scheduler-side adapter, which is what issues lookups."""
    return LMCacheMPSchedulerAdapter(
        server_urls=[server_url],
        context=zmq.Context.instance(),
        model_name="testmodel",
        vllm_block_size=PAGE_SIZE,
        parallel_strategy=parallel_strategy(),
        mq_timeout=DEFAULT_TIMEOUT,
    )


def drive_until_finished(
    adapter: LMCacheMPWorkerAdapter, request_id: str, timeout: float = DEFAULT_TIMEOUT
) -> float:
    """Poll ``get_finished`` the way a worker does, returning when it reports.

    Args:
        adapter: The worker adapter.
        request_id: The retrieve to wait for.
        timeout: Seconds before giving up.

    Returns:
        Seconds from entering the loop to the request being reported.

    Raises:
        TimeoutError: If the request is never reported.
    """
    start = time.perf_counter()
    deadline = start + timeout
    while time.perf_counter() < deadline:
        _, finished_retrieves = adapter.get_finished(set())
        if finished_retrieves and request_id in finished_retrieves:
            return time.perf_counter() - start
        time.sleep(1e-4)
    raise TimeoutError(f"{request_id} was never reported finished")


def store_prefix(
    adapter: LMCacheMPWorkerAdapter, token_ids: list[int], blocks: list[int]
) -> None:
    """Store one prefix and wait for the store to be acknowledged."""
    event = torch_dev.Event(interprocess=True)
    event.record()
    request_id = "store_req"
    adapter.submit_store_request(
        request_id,
        LoadStoreOp(
            token_ids=token_ids, block_ids=[blocks], start=0, end=len(token_ids)
        ),
        event,
    )
    deadline = time.perf_counter() + DEFAULT_TIMEOUT
    while time.perf_counter() < deadline:
        finished_stores, _ = adapter.get_finished({request_id})
        if finished_stores and request_id in finished_stores:
            return
        time.sleep(1e-3)
    raise TimeoutError("store was never reported finished")


def lookup_until_visible(
    scheduler: LMCacheMPSchedulerAdapter,
    request_id: str,
    token_ids: list[int],
    timeout: float = 30.0,
) -> int:
    """Look the prefix up until it is visible, leaving its lock open.

    A retrieve reads what its own lookup unlocked, and an acknowledged store is
    not immediately visible to the next lookup, so this doubles as the
    readiness gate and as the lookup the retrieve rides on.

    Args:
        scheduler: The scheduler-side adapter.
        request_id: The retrieve's request id, so the lock belongs to it.
        token_ids: The prefix to look up.
        timeout: Seconds before giving up.

    Returns:
        Matched token count.

    Raises:
        AssertionError: If the prefix never becomes visible.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        scheduler.maybe_submit_lookup_request(request_id, token_ids)
        hit = None
        poll_until = time.time() + 10.0
        while hit is None and time.time() < poll_until:
            hit = scheduler.check_lookup_result(request_id)
            if hit is None:
                time.sleep(1e-3)
        if hit:
            return hit
        scheduler.cleanup_lookup_result(request_id)
        time.sleep(0.2)
    raise AssertionError("the stored prefix never became visible to lookup")


@pytest.mark.parametrize("layers_per_stage", [0, 1, 8])
def test_worker_releases_early_and_still_reads_landed_kv(
    server_url: str,
    kv_caches: dict[str, torch.Tensor],
    layers_per_stage: int,
):
    """A released request reads correct KV, and layer-major releases sooner.

    The loop below is what a worker runs: submit the retrieve, poll
    ``get_finished`` until the request comes back, then walk the layers calling
    ``wait_for_layer_load`` before reading each one. With layer-major on, the
    request comes back while later layers are still copying, so the per-layer
    wait is the only thing keeping the read correct -- which is exactly what
    the assertions check.
    """
    adapter = make_adapter(server_url, layers_per_stage)
    scheduler = make_scheduler(server_url)
    try:
        adapter.register_kv_caches(kv_caches)

        token_ids = [(layers_per_stage * 1000 + i) % 50000 for i in range(TOKENS)]
        source = list(range(0, BLOCKS))
        store_prefix(adapter, token_ids, source)
        request_id = f"retrieve_{layers_per_stage}"
        lookup_until_visible(scheduler, request_id, token_ids)

        expected = {
            name: tensor[:, :BLOCKS].clone() for name, tensor in kv_caches.items()
        }

        dest_offset = BLOCKS * 2
        dest = list(range(dest_offset, dest_offset + BLOCKS))
        event = torch_dev.Event(interprocess=True)
        event.record()
        adapter.submit_retrieve_request(
            request_id,
            LoadStoreOp(
                token_ids=token_ids, block_ids=[dest], start=0, end=len(token_ids)
            ),
            event,
        )

        released_after = drive_until_finished(adapter, request_id)

        if layers_per_stage > 0:
            assert adapter._arrival_pool is not None, (  # noqa: SLF001
                "layer-major was asked for but the worker built no arrival pool"
            )

        # Walk the layers the way the model does: wait, then read.
        for index in range(NUM_LAYERS):
            adapter.wait_for_layer_load(layer_name(index))
            torch_dev.current_stream().synchronize()
            got = kv_caches[layer_name(index)][:, dest_offset : dest_offset + BLOCKS]
            assert torch.equal(expected[layer_name(index)], got), (
                f"layers_per_stage={layers_per_stage}, layer={index}: the model "
                "would have read KV that had not landed"
            )

        print(
            f"\nlayers_per_stage={layers_per_stage}: released after "
            f"{released_after * 1e3:.1f} ms"
        )
    finally:
        scheduler.shutdown()
        adapter.shutdown()


def test_layer_major_hands_the_request_back_before_the_copy_ends(
    server_url: str,
    kv_caches: dict[str, torch.Tensor],
):
    """Layer-major reports the retrieve while its later layers are still moving.

    This is the claim the feature rests on. It is checked on the adapter's own
    bookkeeping rather than on a stopwatch: a request released early sits in the
    draining set with its transfer future still unresolved, which is a state the
    chunk-major path can never be in.
    """
    adapter = make_adapter(server_url, 1)
    scheduler = make_scheduler(server_url)
    try:
        adapter.register_kv_caches(kv_caches)
        token_ids = [(7777 + i) % 50000 for i in range(TOKENS)]
        store_prefix(adapter, token_ids, list(range(0, BLOCKS)))
        request_id = "early_release"
        lookup_until_visible(scheduler, request_id, token_ids)

        dest_offset = BLOCKS * 2
        event = torch_dev.Event(interprocess=True)
        event.record()
        adapter.submit_retrieve_request(
            request_id,
            LoadStoreOp(
                token_ids=token_ids,
                block_ids=[list(range(dest_offset, dest_offset + BLOCKS))],
                start=0,
                end=len(token_ids),
            ),
            event,
        )

        released_early = False
        deadline = time.perf_counter() + DEFAULT_TIMEOUT
        while time.perf_counter() < deadline:
            _, finished = adapter.get_finished(set())
            if finished and request_id in finished:
                # Released. If the transfer future is still unresolved the
                # release cannot have waited for the whole copy.
                draining = adapter._draining_retrieves  # noqa: SLF001
                released_early = request_id in draining
                break
            time.sleep(1e-4)
        else:
            raise TimeoutError("the retrieve was never reported")

        assert released_early, (
            "the request was reported only once its whole transfer had "
            "finished, so the early release never happened"
        )
    finally:
        scheduler.shutdown()
        adapter.shutdown()
