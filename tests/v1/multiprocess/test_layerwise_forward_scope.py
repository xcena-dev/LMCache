# SPDX-License-Identifier: Apache-2.0
"""A resumed async load must still gate the forward pass without a new retrieve."""

# Standard
from collections.abc import Iterable

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPConnectorMetadata


class RecordingWorker:
    """Record the public per-forward wait scope without CUDA or an MP server."""

    def __init__(self) -> None:
        self.forward_requests: list[list[str]] = []

    def set_forward_requests(self, request_ids: Iterable[str]) -> None:
        """Capture which requests the next attention layers must wait on."""
        self.forward_requests.append(list(request_ids))


@pytest.mark.parametrize("resumed", [["resumed"], ["resumed-a", "resumed-b"]])
def test_resumed_load_waits_without_a_new_retrieve(resumed: list[str]) -> None:
    """The RPC was submitted in an earlier step, so this step has no RETRIEVE."""
    worker = RecordingWorker()
    connector = object.__new__(LMCacheMPConnector)
    connector.worker_adapter = worker
    metadata = LMCacheMPConnectorMetadata()
    metadata.forward_request_ids = resumed
    assert len(metadata) == 0

    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(None)

    assert worker.forward_requests == [resumed]


def test_wait_scope_is_replaced_when_the_forward_batch_changes() -> None:
    """A draining older retrieve must not become a barrier for a later batch."""
    worker = RecordingWorker()
    connector = object.__new__(LMCacheMPConnector)
    connector.worker_adapter = worker

    for request_ids in (["resumed"], ["other"], []):
        metadata = LMCacheMPConnectorMetadata()
        metadata.forward_request_ids = request_ids
        connector.bind_connector_metadata(metadata)
        connector.start_load_kv(None)

    assert worker.forward_requests == [["resumed"], ["other"], []]
