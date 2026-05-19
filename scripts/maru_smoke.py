#!/home/shson/.venv/bin/python
# SPDX-License-Identifier: Apache-2.0
"""Maru integration smoke test.

Run after ``maru-resource-manager`` and ``maru-server`` are up. Use
``--verbose`` to see full stack traces on failure.

The script imports the ``lmcache`` package as the active environment
resolves it — i.e. the version your ``uv pip install . --no-build-
isolation`` last produced. T3 prints the loaded path so a stale
install is obvious; re-run ``uv pip install . --no-build-isolation``
after editing the checkout if you want the smoke to exercise the
new code.

Tiers (each builds on the previous):

    T1. maru's own ``examples/basic/single_instance.py`` runs to
        completion. If this fails the issue is in the maru runtime
        / environment, not in LMCache.
    T2. ``MaruHandler.connect()`` returns control without crashing.
    T3. ``MaruMemoryAllocator`` constructs (lazy — no RPC yet).
    T4. ``MaruMemoryAllocator.init_layout()`` brings the CXL pool up.
    T5. ``MaruMemoryAllocator.batched_allocate()`` returns MemoryObjs.
    T6. ``L1MemoryManager.register_kv_layout()`` forwards down.
    T7. ``MaruL2Adapter`` constructs (lazy — handler stays ``None``).
    T8. First store triggers the lazy ``MaruHandler.connect`` and
        the store → load round-trip preserves bytes.

Usage::

    /home/shson/.venv/bin/python scripts/maru_smoke.py
    /home/shson/.venv/bin/python scripts/maru_smoke.py \
        --server maru://localhost:5555 --pool-gb 1 -v
"""

# Standard
from pathlib import Path
import argparse
import os
import subprocess
import sys
import traceback

PYTHON = sys.executable
MARU_EXAMPLE = Path("/home/shson/maru/examples/basic/single_instance.py")

# ANSI colours, disabled when not a TTY so the script can be piped to a file.
_TTY = sys.stdout.isatty()
GREEN = "\033[32m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def hdr(name: str) -> None:
    print(f"\n{BOLD}=== {name} ==={RESET}", flush=True)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}", flush=True)


def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}", flush=True)


# ---------------------------------------------------------------------------
# T1 — maru's own example as a subprocess, captures the SIGBUS exit code.
# ---------------------------------------------------------------------------


def t1_maru_example() -> bool:
    hdr("T1 — maru built-in single_instance example")
    if not MARU_EXAMPLE.is_file():
        fail(f"example not found: {MARU_EXAMPLE}")
        dim("Adjust MARU_EXAMPLE in this script if maru lives elsewhere.")
        return False

    try:
        proc = subprocess.run(
            [PYTHON, "-u", str(MARU_EXAMPLE)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        fail("timed out after 30s")
        return False

    if proc.returncode == 0:
        ok(f"{MARU_EXAMPLE.name} ran to completion")
        return True

    fail(f"exit code {proc.returncode}")
    if proc.returncode == 135:
        dim("Exit 135 = SIGBUS (128 + 7). Almost certainly an mmap/DAX")
        dim("permission or pool-backing issue on the maru side.")
    dim("---- stdout (tail) ----")
    for line in proc.stdout.splitlines()[-10:]:
        dim(line)
    dim("---- stderr (tail) ----")
    for line in proc.stderr.splitlines()[-10:]:
        dim(line)
    return False


# ---------------------------------------------------------------------------
# T2 — MaruHandler.connect() returns. Run in subprocess so a SIGBUS doesn't
# kill the rest of the smoke run.
# ---------------------------------------------------------------------------


_T2_SNIPPET = r"""
import sys
from maru import MaruConfig, MaruHandler
mc = MaruConfig(
    server_url="{server_url}",
    instance_id="maru-smoke-t2",
    pool_size={pool_bytes},
    chunk_size_bytes=4 * 1024 * 1024,
    auto_connect=False,
    timeout_ms=5000,
)
h = MaruHandler(mc)
print("BUILT", flush=True)
ok = h.connect()
print(f"CONNECT_RETURNED ok={{ok}}", flush=True)
h.close()
print("CLOSE_OK", flush=True)
"""


def _run_in_subprocess(snippet: str, label: str) -> tuple[bool, str]:
    """Return (success, stdout) — success means exit 0."""
    try:
        proc = subprocess.run(
            [PYTHON, "-u", "-c", snippet],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, f"<{label} timed out>"
    success = proc.returncode == 0
    return success, proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")


def t2_handler_connect(server_url: str, pool_bytes: int) -> bool:
    hdr("T2 — MaruHandler.connect()")
    snippet = _T2_SNIPPET.format(server_url=server_url, pool_bytes=pool_bytes)
    success, out = _run_in_subprocess(snippet, "T2")
    last = [line for line in out.splitlines() if line.strip()][-5:]
    if not success:
        fail("connect path crashed or returned non-zero")
        for line in last:
            dim(line)
        return False
    if "CLOSE_OK" in out:
        ok("connect/close cycle clean")
        return True
    fail("subprocess returned 0 but did not reach CLOSE_OK")
    for line in last:
        dim(line)
    return False


# ---------------------------------------------------------------------------
# T3–T6 — LMCache-side. Run in-process now that we know the runtime is sane.
# ---------------------------------------------------------------------------


def t3_allocator_construct() -> bool:
    hdr("T3 — MaruMemoryAllocator __init__ (lazy)")
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
    import lmcache

    # Surface which lmcache we picked up so a path-shadowing surprise
    # (e.g. an older pip-installed copy) is obvious.
    dim(f"lmcache loaded from: {Path(lmcache.__file__).parent}")

    cfg = MaruL1Config(
        server_url="maru://unused-for-this-test:1",
        pool_size_bytes=1,
        instance_id="maru-smoke-t3",
    )
    alloc = MaruMemoryAllocator(cfg)
    if alloc.is_initialized:
        fail("allocator is_initialized=True after __init__ (lazy contract broken)")
        return False
    if alloc._handler is not None or alloc._cxl_adapter is not None:
        fail("handler/adapter populated before init_layout")
        return False
    ok("__init__ returned with handler=adapter=None, is_initialized=False")
    return True


def t4_init_layout(server_url: str, pool_bytes: int) -> bool:
    hdr("T4 — MaruMemoryAllocator.init_layout()")
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
    from lmcache.v1.memory_management import MemoryFormat

    cfg = MaruL1Config(
        server_url=server_url,
        pool_size_bytes=pool_bytes,
        instance_id="maru-smoke-t4",
    )
    alloc = MaruMemoryAllocator(cfg)
    shapes = [torch.Size([2, 32, 256, 128])]  # 4 MiB / chunk
    dtypes = [torch.float16]
    try:
        alloc.init_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256)
    except Exception:
        fail("init_layout raised")
        traceback.print_exc()
        return False
    if not alloc.is_initialized:
        fail("init_layout returned but is_initialized=False")
        return False
    ok(f"init_layout OK (single_token_size={alloc.single_token_size})")

    # Idempotent same-layout
    try:
        alloc.init_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256)
        ok("idempotent same-layout call accepted")
    except Exception:
        fail("idempotent same-layout call raised")
        traceback.print_exc()
        return False

    # Mismatched layout — must reject.
    try:
        alloc.init_layout(
            [torch.Size([2, 16, 256, 128])], dtypes, MemoryFormat.KV_2LTD, 256
        )
        fail("layout-mismatch did NOT raise (single-model constraint broken)")
        return False
    except ValueError:
        ok("layout-mismatch rejected as expected")

    # Stash for T5 by returning the live allocator.
    t4_init_layout._alloc = alloc  # type: ignore[attr-defined]
    return True


def t5_batched_allocate() -> bool:
    hdr("T5 — MaruMemoryAllocator.batched_allocate()")
    alloc = getattr(t4_init_layout, "_alloc", None)
    if alloc is None:
        fail("T4 must run first to populate the allocator")
        return False
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_management import MemoryFormat

    shapes = [torch.Size([2, 32, 256, 128])]
    dtypes = [torch.float16]
    try:
        objs = alloc.batched_allocate(
            shapes, dtypes, batch_size=4, fmt=MemoryFormat.KV_2LTD
        )
    except Exception:
        fail("batched_allocate raised")
        traceback.print_exc()
        return False
    if not objs or len(objs) != 4:
        fail(f"expected 4 MemoryObjs, got {None if not objs else len(objs)}")
        return False
    ok(f"got {len(objs)} MemoryObjs back from the pool")

    alloc.close()
    ok("close() OK")
    return True


def t6_register_kv_layout(server_url: str, pool_bytes: int) -> bool:
    hdr("T6 — L1MemoryManager.register_kv_layout()")
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.config import L1MemoryManagerConfig
    from lmcache.v1.distributed.maru_memory_allocator import MaruL1Config
    from lmcache.v1.distributed.memory_manager import L1MemoryManager
    from lmcache.v1.memory_management import MemoryFormat

    maru_cfg = MaruL1Config(
        server_url=server_url,
        pool_size_bytes=pool_bytes,
        instance_id="maru-smoke-t6",
    )
    mgr = L1MemoryManager(
        L1MemoryManagerConfig(size_in_bytes=0, use_lazy=False, maru_config=maru_cfg)
    )
    try:
        mgr.register_kv_layout(
            [torch.Size([2, 32, 256, 128])],
            [torch.float16],
            MemoryFormat.KV_2LTD,
            256,
        )
    except Exception:
        fail("register_kv_layout raised")
        traceback.print_exc()
        mgr.close()
        return False
    if not mgr._allocator.is_initialized:  # type: ignore[attr-defined]
        fail("register_kv_layout did not initialize the allocator")
        mgr.close()
        return False
    ok("register_kv_layout drove the allocator to init_layout()")
    mgr.close()
    ok("L1MemoryManager.close() OK")
    return True


def t7_l2_adapter_connect(server_url: str, pool_bytes: int) -> bool:
    hdr("T7 — MaruL2Adapter construction (lazy — no RPC yet)")
    # First Party
    from lmcache.v1.distributed.l2_adapters.maru_l2_adapter import (
        MaruL2Adapter,
        MaruL2AdapterConfig,
    )

    chunk_size_bytes = 1 << 20  # 1 MiB — small enough for a quick round trip
    cfg = MaruL2AdapterConfig(
        server_url=server_url,
        pool_size_gb=max(pool_bytes / (1 << 30), 0.125),
        chunk_size_bytes=chunk_size_bytes,
        instance_id="maru-smoke-t7",
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )
    try:
        adapter = MaruL2Adapter(cfg)
    except Exception:
        fail("MaruL2Adapter construction raised")
        traceback.print_exc()
        return False
    fds = {
        adapter.get_store_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
        adapter.get_load_event_fd(),
    }
    if len(fds) != 3:
        fail(f"expected 3 distinct event fds, got {len(fds)}")
        adapter.close()
        return False
    if adapter._handler is not None:
        fail("handler populated before any store (lazy contract broken)")
        adapter.close()
        return False
    ok(f"constructed; handler=None, event fds = {sorted(fds)}")

    t7_l2_adapter_connect._adapter = adapter  # type: ignore[attr-defined]
    t7_l2_adapter_connect._chunk_size_bytes = chunk_size_bytes  # type: ignore[attr-defined]
    return True


def t8_l2_store_load_roundtrip() -> bool:
    hdr("T8 — first store triggers lazy connect + store → load round-trip")
    # Standard
    from unittest import mock
    import time

    # Third Party
    import numpy as np

    # First Party
    from lmcache.native_storage_ops import (  # noqa: F401 — surface import errors here
        Bitmap,
    )
    from lmcache.v1.distributed.api import ObjectKey

    adapter = getattr(t7_l2_adapter_connect, "_adapter", None)
    chunk_size_bytes = getattr(t7_l2_adapter_connect, "_chunk_size_bytes", None)
    if adapter is None or chunk_size_bytes is None:
        fail("T7 must run first to populate the adapter")
        return False

    # 1) Set up a deterministic byte pattern in a DRAM-side numpy
    # buffer and wrap it as a fake MemoryObj (data_ptr + get_size).
    payload_bytes = min(chunk_size_bytes, 65536)
    src_arr = np.frombuffer(
        bytes(i % 256 for i in range(payload_bytes)), dtype=np.uint8
    ).copy()  # writable copy

    src_obj = mock.MagicMock(name="DramMemoryObj")
    src_obj.data_ptr = int(src_arr.ctypes.data)
    src_obj.get_size = mock.MagicMock(return_value=payload_bytes)

    key = ObjectKey(
        chunk_hash=(0xCAFEBABE).to_bytes(4, "big"),
        model_name="smoke-t8",
        kv_rank=0xABCD,
        cache_salt="",
    )

    # 2) Store: alloc CXL page + memcpy DRAM→CXL + batch_store.
    store_task = adapter.submit_store_task([key], [src_obj])
    deadline = time.time() + 10
    while time.time() < deadline:
        completed = adapter.pop_completed_store_tasks()
        if store_task in completed:
            break
        time.sleep(0.05)
    else:
        fail("store task did not complete within 10s")
        return False
    if not completed[store_task]:
        fail("store task returned failure")
        return False
    ok(f"stored {payload_bytes} bytes")

    # 3) Lookup + lock — bit 0 should be set after batch_pin.
    lookup_task = adapter.submit_lookup_and_lock_task([key])
    bm = None
    deadline = time.time() + 10
    while time.time() < deadline:
        bm = adapter.query_lookup_and_lock_result(lookup_task)
        if bm is not None:
            break
        time.sleep(0.05)
    if bm is None or not bm.test(0):
        fail("lookup did not report a hit for the just-stored key")
        return False
    ok("lookup hit")

    # 4) Load into a fresh DRAM buffer and verify bytes match.
    dst_arr = np.zeros(payload_bytes, dtype=np.uint8)
    dst_obj = mock.MagicMock(name="DramMemoryObj")
    dst_obj.data_ptr = int(dst_arr.ctypes.data)
    dst_obj.get_size = mock.MagicMock(return_value=payload_bytes)

    load_task = adapter.submit_load_task([key], [dst_obj])
    bm = None
    deadline = time.time() + 10
    while time.time() < deadline:
        bm = adapter.query_load_result(load_task)
        if bm is not None:
            break
        time.sleep(0.05)
    if bm is None or not bm.test(0):
        fail("load did not report success for the just-stored key")
        return False
    if not np.array_equal(src_arr, dst_arr):
        fail("loaded bytes do not match the source")
        return False
    ok(f"loaded {payload_bytes} bytes; DRAM↔CXL↔DRAM round-trip preserved")

    # 5) Cleanup: unlock + delete + close so the next run starts cold.
    adapter.submit_unlock([key])
    adapter.delete([key])
    adapter.close()
    ok("unlock + delete + close OK")
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=os.environ.get("MARU_SERVER_URL", "maru://localhost:5555"),
        help="MaruServer URL (default: maru://localhost:5555).",
    )
    parser.add_argument(
        "--pool-gb",
        type=float,
        default=1.0,
        help="CXL pool size in GB to request (default: 1).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Currently a no-op placeholder; tracebacks are always printed on failure.",
    )
    args = parser.parse_args()
    pool_bytes = int(args.pool_gb * (1 << 30))

    print(f"server={args.server}  pool={args.pool_gb} GB", flush=True)

    # T1, T2 are subprocess-isolated so SIGBUS doesn't terminate us.
    if not t1_maru_example():
        print(f"\n{RED}stop:{RESET} fix the maru runtime before continuing.")
        return 1
    if not t2_handler_connect(args.server, pool_bytes):
        print(f"\n{RED}stop:{RESET} MaruHandler.connect() not stable.")
        return 1

    # T3–T8 run in-process — at this point we trust the runtime.
    # T7/T8 cover the L2 adapter path (DRAM→CXL store + CXL→DRAM load
    # via ``MaruL2Adapter``); they share state through the
    # ``t7_l2_adapter_connect._adapter`` stash so T8 reuses the
    # already-connected handler.
    results = [
        t3_allocator_construct(),
        t4_init_layout(args.server, pool_bytes),
        t5_batched_allocate(),
        t6_register_kv_layout(args.server, pool_bytes),
        t7_l2_adapter_connect(args.server, pool_bytes),
        t8_l2_store_load_roundtrip(),
    ]
    if all(results):
        print(f"\n{GREEN}{BOLD}all tiers passed.{RESET}")
        return 0
    print(f"\n{RED}{BOLD}some tiers failed (see above).{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
