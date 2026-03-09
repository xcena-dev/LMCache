# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.maru_connector import (
    MaruConnector,
    MaruConnectorConfig,
)

logger = init_logger(__name__)


class MaruConnectorAdapter(ConnectorAdapter):
    """Adapter for maru scheme."""

    def __init__(self) -> None:
        super().__init__("maru://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Validate URL format (requires host:port)
        _ = parse_remote_url(context.url)

        # Parse configuration from URL
        maru_config = MaruConnectorConfig.from_url(context.url)
        logger.info(
            "Maru config from URL: server_url=%s, pool_size=%d",
            maru_config.server_url,
            maru_config.pool_size,
        )

        if context.config is None or context.metadata is None:
            raise ValueError("Maru connector requires config and metadata")

        return MaruConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            metadata=context.metadata,
        )
