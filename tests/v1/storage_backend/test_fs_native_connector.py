# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ FS connector (``lmcache.lmcache_fs``).

Focus: the O_DIRECT alignment guard. O_DIRECT requires the file offset,
the transfer length, and the user buffer address to all be block-aligned.
A destination buffer that is not block-aligned must fall back to buffered
I/O instead of failing the read() with EINVAL (which previously broke all
L2 GETs when the L1 pool base was only 64-byte aligned).
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import mmap
import os
import time

# Third Party
import pytest

if TYPE_CHECKING:
    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")

BLOCK_SIZE = 4096
# Key format expected by FSConnector::key_to_filename:
# <model_name>@<kv_rank_hex>@<chunk_hash_hex>
TEST_KEY = "test-model@00000000@" + "ab" * 32


def _wait_completion(
    client: "LMCacheFSClient", future_id: int, timeout: float = 10.0
) -> tuple[bool, str]:
    """Poll drain_completions until the given future completes.

    Args:
        client: The native FS client.
        future_id: Future id returned by a submit_batch_* call.
        timeout: Maximum seconds to wait.

    Returns:
        The (ok, error) pair of the matching completion.

    Raises:
        TimeoutError: If the completion does not arrive within the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for fid, ok, error, _result_bools in client.drain_completions():
            if fid == future_id:
                return ok, error
        time.sleep(0.01)
    raise TimeoutError(f"future {future_id} did not complete within {timeout}s")


def _page_aligned_view(size: int, offset: int) -> memoryview:
    """Return a memoryview of `size` bytes starting `offset` bytes past a
    page boundary (mmap always returns page-aligned memory)."""
    pool = mmap.mmap(-1, size + mmap.PAGESIZE)
    return memoryview(pool)[offset : offset + size]


def _filesystem_supports_odirect(directory: Path) -> bool:
    """Return whether files in `directory` can be opened with O_DIRECT
    (e.g. tmpfs does not support it)."""
    probe = directory / "odirect_probe.bin"
    probe.write_bytes(b"\0" * BLOCK_SIZE)
    try:
        fd = os.open(probe, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return False
    finally:
        probe.unlink()
    os.close(fd)
    return True


def _roundtrip(
    client: "LMCacheFSClient", buffer_offset: int, key: str = TEST_KEY
) -> None:
    """Write a payload through the client and read it back, using buffers
    that start `buffer_offset` bytes past a page boundary.

    Args:
        client: The native FS client.
        buffer_offset: Bytes past a page boundary at which buffers start.
        key: Cache key to store under. Must not have been stored before:
            set skips keys that already exist on disk (content-addressed).
    """
    size = BLOCK_SIZE * 2  # block-multiple length: only the address may gate
    payload = os.urandom(size)

    src = _page_aligned_view(size, buffer_offset)
    src[:] = payload
    ok, error = _wait_completion(client, client.submit_batch_set([key], [src]))
    assert ok, f"set failed: {error}"

    dst = _page_aligned_view(size, buffer_offset)
    ok, error = _wait_completion(client, client.submit_batch_get([key], [dst]))
    assert ok, f"get failed: {error}"
    assert bytes(dst) == payload


@pytest.fixture
def odirect_client(tmp_path):
    """A native FS client with use_odirect enabled on a temp directory."""
    client = lmcache_fs.LMCacheFSClient(
        str(tmp_path), 1, "", True, 0
    )  # base_path, num_workers, relative_tmp_dir, use_odirect, read_ahead_size
    yield client
    client.close()


def test_odirect_unaligned_buffer_falls_back_to_buffered(odirect_client, capfd):
    """A 64-byte-aligned buffer (the torch.empty CPU base alignment) must not
    make O_DIRECT reads fail with EINVAL; the guard falls back to buffered
    I/O and the roundtrip succeeds on any filesystem/device. The fallback
    warning is emitted once per worker connection, not once per request."""
    _roundtrip(odirect_client, buffer_offset=64)
    _roundtrip(odirect_client, buffer_offset=64, key="test-model@00000000@" + "cd" * 32)
    assert capfd.readouterr().err.count("falling back to buffered I/O") == 1


def test_odirect_aligned_buffer_roundtrip(odirect_client, tmp_path, capfd):
    """A page-aligned buffer keeps the real O_DIRECT path and still
    roundtrips correctly, without emitting the fallback warning."""
    if not _filesystem_supports_odirect(tmp_path):
        pytest.skip("filesystem does not support O_DIRECT (e.g. tmpfs)")
    _roundtrip(odirect_client, buffer_offset=0)
    assert "falling back to buffered I/O" not in capfd.readouterr().err
