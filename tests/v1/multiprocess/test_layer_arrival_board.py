# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.layer_arrival_board import (
    SLOT_BYTES,
    LayerArrivalBoard,
)


def _name(tag: str) -> str:
    return f"lmc_test_arrival_{tag}_{os.getpid()}"


@pytest.fixture
def board():
    b = LayerArrivalBoard.create(_name("main"), num_slots=8)
    try:
        yield b
    finally:
        b.close()


def test_slots_start_empty(board):
    assert [board.read(i) for i in range(board.num_slots)] == [0] * 8


def test_publish_then_read(board):
    board.publish(3, 1)
    assert board.read(3) == 1
    board.publish(3, 7)
    assert board.read(3) == 7


def test_slots_are_independent(board):
    for slot in range(board.num_slots):
        board.publish(slot, slot + 1)
    assert [board.read(s) for s in range(board.num_slots)] == list(range(1, 9))


def test_slots_do_not_share_a_cache_line():
    assert SLOT_BYTES == 64


def test_reset_clears_one_slot(board):
    board.publish(2, 5)
    board.publish(3, 6)
    board.reset(2)
    assert board.read(2) == 0
    assert board.read(3) == 6


@pytest.mark.parametrize("slot", [-1, 8, 99])
def test_out_of_range_slot_is_rejected(board, slot):
    with pytest.raises(IndexError):
        board.read(slot)
    with pytest.raises(IndexError):
        board.publish(slot, 1)


def test_negative_count_is_rejected(board):
    with pytest.raises(ValueError):
        board.publish(0, -1)


def test_num_slots_must_be_positive():
    with pytest.raises(ValueError):
        LayerArrivalBoard.create(_name("bad"), num_slots=0)


def test_creating_the_same_segment_twice_fails(board):
    with pytest.raises(OSError):
        LayerArrivalBoard.create(board.name, num_slots=8)


_WRITER_PROGRAM = """
import sys
import time
from lmcache.v1.multiprocess.layer_arrival_board import LayerArrivalBoard
name = sys.argv[1]
num_slots = int(sys.argv[2])
slot = int(sys.argv[3])
slices = int(sys.argv[4])
board = LayerArrivalBoard.attach(name, num_slots)
try:
    for i in range(slices):
        time.sleep(0.01)
        board.publish(slot, i + 1)
finally:
    board.close()
"""


def test_a_second_process_sees_progress_as_it_happens(board):
    """The point of the board: the reader observes the writer's steps live.

    The writer runs as a plain interpreter rather than a multiprocessing child
    so it never imports this test package, which keeps the child independent of
    the parent's test-time environment.
    """
    slot, slices = 5, 8
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _WRITER_PROGRAM,
            board.name,
            str(board.num_slots),
            str(slot),
            str(slices),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        observed = []
        deadline = time.monotonic() + 60
        last = 0
        while last < slices and time.monotonic() < deadline:
            value = board.read(slot)
            if value != last:
                observed.append(value)
                last = value
            if proc.poll() is not None and board.read(slot) == last == 0:
                break
        _, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, f"writer failed: {err}"
        assert last == slices, f"only reached {last} of {slices}: {err}"
        # Monotonic and gap-free: every step was visible, none went backwards.
        assert observed == list(range(1, slices + 1))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_attach_does_not_clear_existing_progress(board):
    board.publish(1, 4)
    attached = LayerArrivalBoard.attach(board.name, board.num_slots)
    try:
        assert attached.read(1) == 4
    finally:
        attached.close()


def test_close_is_idempotent():
    b = LayerArrivalBoard.create(_name("idem"), num_slots=2)
    b.close()
    b.close()
