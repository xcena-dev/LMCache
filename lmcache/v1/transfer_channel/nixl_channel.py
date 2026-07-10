# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import asyncio
import threading
import time
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryObj,
)

if TYPE_CHECKING:
    # Third Party
    from nixl._api import NixlAgent

# First Party
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""

    pass


class NixlInitRequest(NixlMsgBase):
    local_meta_bytes: bytes  # Metadata from the sender nixl agent
    local_meta_bytes_list: Optional[list[bytes]] = None


class NixlMemRegRequest(NixlMsgBase):
    remote_agent_name: bytes
    local_id: str
    local_xfer_dlist_bytes: bytes
    remote_agent_names: Optional[list[bytes]] = None
    local_xfer_dlist_bytes_list: Optional[list[bytes]] = None


class NixlInitResponse(NixlMsgBase):
    remote_agent_name: bytes
    remote_meta_bytes: bytes  # Metadata from the receiver nixl agent
    remote_agent_names: Optional[list[bytes]] = None
    remote_meta_bytes_list: Optional[list[bytes]] = None


class NixlMemRegResponse(NixlMsgBase):
    remote_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver
    remote_xfer_dlist_bytes_list: Optional[list[bytes]] = None


NixlMsg = Union[
    NixlInitRequest, NixlInitResponse, NixlMemRegRequest, NixlMemRegResponse
]


def _stringify_params(params: Optional[dict[str, Any]]) -> dict[str, str]:
    if not params:
        return {}
    return {str(key): str(value) for key, value in params.items() if value is not None}


def _params_for_backend(
    backend_params: Optional[dict[str, Any]],
    backend: str,
) -> dict[str, str]:
    if not backend_params:
        return {}
    params = {
        key: value
        for key, value in backend_params.items()
        if not isinstance(value, dict)
    }
    nested = backend_params.get(backend)
    if isinstance(nested, dict):
        params.update(nested)
    return _stringify_params(params)


def _merge_ucx_devices(
    backend_params: Optional[dict[str, Any]],
    ucx_devices: str,
) -> dict[str, Any]:
    params = dict(backend_params or {})
    ucx_params = dict(params.get("UCX", {})) if isinstance(params.get("UCX"), dict) else {}
    ucx_params["ucx_devices"] = ucx_devices
    # NIXL's UCX plugin exposes "ucx_devices" via get_plugin_params(), but its
    # engine constructor restricts UCX NET_DEVICES from "device_list". The plugin
    # appends ":1" internally, so keep device_list as bare HCA names.
    ucx_params.setdefault(
        "device_list",
        ", ".join(dev.split(":", 1)[0].strip() for dev in ucx_devices.split(",")),
    )
    params["UCX"] = ucx_params
    return params


class NixlChannel(BaseTransferChannel):
    def __init__(
        self,
        async_mode: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        if "backends" in kwargs:
            backends = kwargs["backends"]
        else:
            backends = ["UCX"]

        self.role = kwargs["role"]

        self.nixl_wrappers = self._create_nixl_wrappers(
            buffer_ptr=kwargs["buffer_ptr"],
            buffer_size=kwargs["buffer_size"],
            page_size=kwargs["align_bytes"],
            tp_rank=kwargs["tp_rank"],
            backends=backends,
            device=device,
            backend_params=kwargs.get("backend_params", None),
            agent_backend_params=kwargs.get("agent_backend_params", None),
            ucx_devices_by_agent=kwargs.get("ucx_devices_by_agent", None),
            num_agents=kwargs.get("num_agents", None),
        )
        self.nixl_wrapper = self.nixl_wrappers[0]
        self.nixl_agent = self.nixl_wrapper.agent

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.remote_xfer_handlers_dict: dict[
            str, list[NixlAgent.nixl_prepped_dlist_handle]
        ] = {}

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self._init_side_channels()

    def _create_nixl_wrappers(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: list[str],
        device: Optional[str],
        backend_params: Optional[dict[str, Any]],
        agent_backend_params: Optional[list[dict[str, Any]]],
        ucx_devices_by_agent: Optional[list[str]],
        num_agents: Optional[int],
    ) -> list["NixlAgentWrapper"]:
        inferred_agents = max(
            1,
            len(agent_backend_params or []),
            len(ucx_devices_by_agent or []),
        )
        agent_count = int(num_agents) if num_agents is not None else inferred_agents
        if agent_count < 1:
            raise ValueError("num_agents must be >= 1")

        wrappers = []
        for agent_idx in range(agent_count):
            params = backend_params
            if agent_backend_params and agent_idx < len(agent_backend_params):
                params = agent_backend_params[agent_idx]
            if ucx_devices_by_agent and agent_idx < len(ucx_devices_by_agent):
                params = _merge_ucx_devices(params, ucx_devices_by_agent[agent_idx])
            wrappers.append(
                NixlAgentWrapper(
                    agent_idx=agent_idx,
                    buffer_ptr=buffer_ptr,
                    buffer_size=buffer_size,
                    page_size=page_size,
                    tp_rank=tp_rank,
                    backends=backends,
                    device=device,
                    backend_params=params,
                )
            )

        if len(wrappers) > 1:
            logger.info("Initialized %d NIXL agents for P2P sharding", len(wrappers))
        return wrappers

    ############################################################
    # Initialization functions
    ############################################################
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for nixl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        local_meta_bytes_list = [
            wrapper.agent.get_agent_metadata() for wrapper in self.nixl_wrappers
        ]
        # Build and send init request
        nixl_init_req = NixlInitRequest(
            local_meta_bytes=local_meta_bytes_list[0],
            local_meta_bytes_list=local_meta_bytes_list,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))

        # Wait remote agent metadata and register remote agent
        nixl_init_resp_bytes = init_tmp_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)
        remote_meta_bytes_list = nixl_init_resp.remote_meta_bytes_list or [
            nixl_init_resp.remote_meta_bytes
        ]
        if len(remote_meta_bytes_list) != len(self.nixl_wrappers):
            raise RuntimeError(
                "NIXL agent count mismatch during peer initialization: "
                f"local={len(self.nixl_wrappers)} remote={len(remote_meta_bytes_list)}"
            )
        remote_agent_names = [
            wrapper.agent.add_remote_agent(remote_meta_bytes_list[idx])
            for idx, wrapper in enumerate(self.nixl_wrappers)
        ]
        logger.info(
            "NIXL peer init local_id=%s peer_id=%s agents=%d remote_agent_names=%s",
            local_id,
            peer_id,
            len(self.nixl_wrappers),
            [str(name) for name in remote_agent_names],
        )

        # Register remote memory
        local_xfer_dlist_bytes_list = [
            wrapper.agent.get_serialized_descs(wrapper.xfer_descs)
            for wrapper in self.nixl_wrappers
        ]
        peer_agent_names = nixl_init_resp.remote_agent_names or [
            nixl_init_resp.remote_agent_name
        ]
        nixl_mem_reg_req = NixlMemRegRequest(
            remote_agent_name=peer_agent_names[0],
            local_id=local_id,
            local_xfer_dlist_bytes=local_xfer_dlist_bytes_list[0],
            remote_agent_names=peer_agent_names,
            local_xfer_dlist_bytes_list=local_xfer_dlist_bytes_list,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )

        remote_xfer_dlist_bytes_list = (
            nixl_mem_reg_resp.remote_xfer_dlist_bytes_list
            or [nixl_mem_reg_resp.remote_xfer_dlist_bytes]
        )
        remote_xfer_handlers = []
        for idx, wrapper in enumerate(self.nixl_wrappers):
            remote_xfer_dlist = wrapper.agent.deserialize_descs(
                remote_xfer_dlist_bytes_list[idx]
            )
            remote_xfer_handlers.append(
                wrapper.agent.prep_xfer_dlist(
                    remote_agent_names[idx], remote_xfer_dlist
                )
            )
        self.remote_xfer_handlers_dict[peer_id] = remote_xfer_handlers
        logger.info(
            "NIXL stored remote xfer handlers peer_id=%s agents=%d",
            peer_id,
            len(remote_xfer_handlers),
        )

        # Send side message if any
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for nixl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        local_meta_bytes_list = [
            wrapper.agent.get_agent_metadata() for wrapper in self.nixl_wrappers
        ]
        # Build and send init request
        nixl_init_req = NixlInitRequest(
            local_meta_bytes=local_meta_bytes_list[0],
            local_meta_bytes_list=local_meta_bytes_list,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))
        # Wait remote agent metadata and register remote agent
        nixl_init_resp_bytes = await init_tmp_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)
        remote_meta_bytes_list = nixl_init_resp.remote_meta_bytes_list or [
            nixl_init_resp.remote_meta_bytes
        ]
        if len(remote_meta_bytes_list) != len(self.nixl_wrappers):
            raise RuntimeError(
                "NIXL agent count mismatch during peer initialization: "
                f"local={len(self.nixl_wrappers)} remote={len(remote_meta_bytes_list)}"
            )
        remote_agent_names = [
            wrapper.agent.add_remote_agent(remote_meta_bytes_list[idx])
            for idx, wrapper in enumerate(self.nixl_wrappers)
        ]
        logger.info(
            "NIXL async peer init local_id=%s peer_id=%s agents=%d remote_agent_names=%s",
            local_id,
            peer_id,
            len(self.nixl_wrappers),
            [str(name) for name in remote_agent_names],
        )

        # Register remote memory
        local_xfer_dlist_bytes_list = [
            wrapper.agent.get_serialized_descs(wrapper.xfer_descs)
            for wrapper in self.nixl_wrappers
        ]
        peer_agent_names = nixl_init_resp.remote_agent_names or [
            nixl_init_resp.remote_agent_name
        ]
        nixl_mem_reg_req = NixlMemRegRequest(
            remote_agent_name=peer_agent_names[0],
            local_id=local_id,
            local_xfer_dlist_bytes=local_xfer_dlist_bytes_list[0],
            remote_agent_names=peer_agent_names,
            local_xfer_dlist_bytes_list=local_xfer_dlist_bytes_list,
        )

        await init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = await init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )

        remote_xfer_dlist_bytes_list = (
            nixl_mem_reg_resp.remote_xfer_dlist_bytes_list
            or [nixl_mem_reg_resp.remote_xfer_dlist_bytes]
        )
        remote_xfer_handlers = []
        for idx, wrapper in enumerate(self.nixl_wrappers):
            remote_xfer_dlist = wrapper.agent.deserialize_descs(
                remote_xfer_dlist_bytes_list[idx]
            )
            remote_xfer_handlers.append(
                wrapper.agent.prep_xfer_dlist(
                    remote_agent_names[idx], remote_xfer_dlist
                )
            )
        self.remote_xfer_handlers_dict[peer_id] = remote_xfer_handlers
        logger.info(
            "NIXL stored async remote xfer handlers peer_id=%s agents=%d",
            peer_id,
            len(remote_xfer_handlers),
        )

        # Send side message if any
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_xfer_handlers_dict

    def _init_side_channels(self):
        if self.peer_init_url is None:
            return

        if self.async_mode:
            # Start listening coroutine for initialization side channel
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            # Start listening thread for initialization side channel
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[NixlMsg, InitSideMsgBase]
    ) -> Union[NixlMsg, InitSideRetMsgBase]:
        resp: Union[NixlMsg, InitSideRetMsgBase]
        if isinstance(req, NixlInitRequest):
            local_meta_bytes_list = req.local_meta_bytes_list or [req.local_meta_bytes]
            if len(local_meta_bytes_list) != len(self.nixl_wrappers):
                raise RuntimeError(
                    "NIXL agent count mismatch during init handling: "
                    f"local={len(self.nixl_wrappers)} remote={len(local_meta_bytes_list)}"
                )
            agent_names = [
                wrapper.agent.add_remote_agent(local_meta_bytes_list[idx])
                for idx, wrapper in enumerate(self.nixl_wrappers)
            ]
            logger.info(
                "NIXL init request handled agents=%d remote_agent_names=%s",
                len(self.nixl_wrappers),
                [str(name) for name in agent_names],
            )
            local_response_meta_bytes_list = [
                wrapper.agent.get_agent_metadata() for wrapper in self.nixl_wrappers
            ]

            resp = NixlInitResponse(
                remote_agent_name=agent_names[0],
                remote_meta_bytes=local_response_meta_bytes_list[0],
                remote_agent_names=agent_names,
                remote_meta_bytes_list=local_response_meta_bytes_list,
            )

            logger.info("Replying initialization response")

        elif isinstance(req, NixlMemRegRequest):
            local_xfer_descs_list = [
                wrapper.agent.get_serialized_descs(wrapper.xfer_descs)
                for wrapper in self.nixl_wrappers
            ]

            remote_agent_names = req.remote_agent_names or [req.remote_agent_name]
            remote_xfer_dlist_bytes_list = req.local_xfer_dlist_bytes_list or [
                req.local_xfer_dlist_bytes
            ]
            if len(remote_xfer_dlist_bytes_list) != len(self.nixl_wrappers):
                raise RuntimeError(
                    "NIXL agent count mismatch during memory registration: "
                    f"local={len(self.nixl_wrappers)} "
                    f"remote={len(remote_xfer_dlist_bytes_list)}"
                )

            remote_xfer_handlers = []
            for idx, wrapper in enumerate(self.nixl_wrappers):
                remote_xfer_dlist = wrapper.agent.deserialize_descs(
                    remote_xfer_dlist_bytes_list[idx]
                )
                remote_xfer_handlers.append(
                    wrapper.agent.prep_xfer_dlist(
                        remote_agent_names[idx], remote_xfer_dlist
                    )
                )
            self.remote_xfer_handlers_dict[req.local_id] = remote_xfer_handlers
            logger.info(
                "NIXL mem registration handled local_id=%s agents=%d remote_agent_names=%s",
                req.local_id,
                len(remote_xfer_handlers),
                [str(name) for name in remote_agent_names],
            )

            resp = NixlMemRegResponse(
                remote_xfer_dlist_bytes=local_xfer_descs_list[0],
                remote_xfer_dlist_bytes_list=local_xfer_descs_list,
            )

            logger.info("Replying mem register response")
        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        # NOTE: Initialization has to be two stages:
        # (1) Exchanging the metadata.
        # (2) Registering the memory descriptors.
        # Otherwise, there's a chance that nixl got stuck
        # (handle always give "PROC" status) during the first request.
        # (3) Exchanging side messages if any. This depends on the backend
        # that uses the channel.
        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[NixlMsg, SideMsg])

                resp = self._handle_init_msg(req)

                self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)
        logger.info("Starting async initialization loop")

        while self.running:
            try:
                req_bytes = await self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[NixlMsg, SideMsg])

                resp = self._handle_init_msg(req)

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    ############################################################
    # Utility functions
    ############################################################

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in NIXL channel"
            )
        return local_indices

    def _make_prepped_xfer_handles(
        self,
        operation: str,
        objects: Union[list[bytes], list[MemoryObj]],
        remote_handlers: list[Any],
        remote_indexes: list[int],
    ) -> list[tuple["NixlAgentWrapper", Any]]:
        local_indices = self.get_local_mem_indices(objects)
        if len(local_indices) != len(remote_indexes):
            raise RuntimeError(
                "NIXL transfer index mismatch: "
                f"local={len(local_indices)} remote={len(remote_indexes)}"
            )
        if len(remote_handlers) != len(self.nixl_wrappers):
            raise RuntimeError(
                "NIXL remote handler count mismatch: "
                f"local={len(self.nixl_wrappers)} remote={len(remote_handlers)}"
            )

        grouped_local: list[list[int]] = [[] for _ in self.nixl_wrappers]
        grouped_remote: list[list[int]] = [[] for _ in self.nixl_wrappers]
        for idx, (local_index, remote_index) in enumerate(
            zip(local_indices, remote_indexes)
        ):
            agent_idx = idx % len(self.nixl_wrappers)
            grouped_local[agent_idx].append(local_index)
            grouped_remote[agent_idx].append(remote_index)

        handles = []
        for agent_idx, wrapper in enumerate(self.nixl_wrappers):
            if not grouped_local[agent_idx]:
                continue
            logger.info(
                "NIXL transfer plan op=%s agent=%d local_indexes=%s remote_indexes=%s",
                operation,
                wrapper.agent_idx,
                grouped_local[agent_idx],
                grouped_remote[agent_idx],
            )
            handles.append(
                (
                    wrapper,
                    wrapper.agent.make_prepped_xfer(
                        operation,
                        wrapper.xfer_handler,
                        grouped_local[agent_idx],
                        remote_handlers[agent_idx],
                        grouped_remote[agent_idx],
                    ),
                )
            )
        return handles

    def _wait_xfer_handles(self, handles: list[tuple["NixlAgentWrapper", Any]]) -> None:
        wait_time = 0.001
        pending = list(handles)
        while pending:
            next_pending = []
            for wrapper, handle in pending:
                try:
                    status = wrapper.agent.check_xfer_state(handle)
                except Exception:
                    logger.exception(
                        "Exception while checking NIXL transfer status for agent=%d",
                        wrapper.agent_idx,
                    )
                    raise
                logger.debug(f"Transfer status: {status}")

                if status == "ERR":
                    logger.error(
                        "Error in NIXL transfer operation for agent=%d",
                        wrapper.agent_idx,
                    )
                    raise RuntimeError("Failed to transfer objects to remote peer")
                if status == "PROC":
                    next_pending.append((wrapper, handle))
                    continue
                assert status == "DONE", f"Transfer status is {status}, expected DONE"
            pending = next_pending
            if pending:
                time.sleep(wait_time)

    async def _async_wait_xfer_handles(
        self, handles: list[tuple["NixlAgentWrapper", Any]]
    ) -> None:
        wait_time = 0.001
        pending = list(handles)
        while pending:
            next_pending = []
            for wrapper, handle in pending:
                try:
                    status = wrapper.agent.check_xfer_state(handle)
                except Exception:
                    logger.exception(
                        "Exception while checking async NIXL transfer status for agent=%d",
                        wrapper.agent_idx,
                    )
                    raise
                logger.debug(f"Transfer status: {status}")

                if status == "ERR":
                    logger.error(
                        "Error in async NIXL transfer operation for agent=%d",
                        wrapper.agent_idx,
                    )
                    raise RuntimeError("Failed to transfer objects to remote peer")
                if status == "PROC":
                    next_pending.append((wrapper, handle))
                    continue
                assert status == "DONE", f"Transfer status is {status}, expected DONE"
            pending = next_pending
            if pending:
                await asyncio.sleep(wait_time)

    ############################################################
    # Send/Recv functions
    ############################################################

    ### Send and Recv must be called in pair ###
    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    ############################################################
    # Read/Write functions
    ############################################################

    ### Read and Write only need to be called on one side ###
    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the nixl channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None

        handles = self._make_prepped_xfer_handles(
            "WRITE",
            objects,
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )

        for wrapper, handle in handles:
            wrapper.agent.transfer(handle)
        self._wait_xfer_handles(handles)

        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
            Should contain 'receiver_id' and 'remote_indexes'.

        :return: Number of successfully transferred objects.
        """

        assert transfer_spec is not None

        handles = self._make_prepped_xfer_handles(
            "WRITE",
            objects,
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )
        for wrapper, handle in handles:
            wrapper.agent.transfer(handle)
        await self._async_wait_xfer_handles(handles)
        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """

        assert transfer_spec is not None

        handles = self._make_prepped_xfer_handles(
            "READ",
            buffers,
            self.remote_xfer_handlers_dict[transfer_spec["sender_id"]],
            transfer_spec["remote_indexes"],
        )
        for wrapper, handle in handles:
            wrapper.agent.transfer(handle)
        await self._async_wait_xfer_handles(handles)
        return len(buffers)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        for wrapper in self.nixl_wrappers:
            wrapper.agent.deregister_memory(wrapper.reg_descs)
            wrapper.agent.release_dlist_handle(wrapper.xfer_handler)

        for remote_xfer_handlers in self.remote_xfer_handlers_dict.values():
            for idx, remote_xfer_handler in enumerate(remote_xfer_handlers):
                self.nixl_wrappers[idx].agent.release_dlist_handle(remote_xfer_handler)


@dataclass
class NixlAgentWrapper:
    agent_idx: int
    agent: "NixlAgent"
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        agent_idx: int,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: list[str],
        device: Optional[str] = None,
        backend_params: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.
            tp_rank (int): The tensor parallel rank.
            backends (list[str]): The list of backends to use.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        self.agent_idx = agent_idx
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        # Handle None backends by setting default to ["UCX"]
        if backends is None:
            backends = ["UCX"]

        # Create a NIXL agent. Backend init params are required for UCX device
        # pinning, so manually instantiate backends when params are supplied.
        if backend_params:
            nixl_agent = NixlAgent(
                str(uuid.uuid4()),
                nixl_agent_config(backends=[]),
            )
            for backend in backends:
                init_params = _params_for_backend(backend_params, backend)
                nixl_agent.create_backend(
                    backend,
                    init_params,
                )
                logger.info(
                    "Created NIXL backend %s with init params %s, effective params %s",
                    backend,
                    init_params,
                    nixl_agent.get_backend_params(backend),
                )
        else:
            nixl_agent = NixlAgent(
                str(uuid.uuid4()),
                nixl_agent_config(backends=backends),
            )

        # Register the memory
        # The four fields are (base_addr, length, dev_id, meta_info)
        # https://github.com/ai-dynamo/nixl/blob/main/src/api/cpp/nixl_descriptors.h#L152
        memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
        mem_type = "cpu" if device == "cpu" else "cuda"

        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type=mem_type)
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, tp_rank))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)
        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler
