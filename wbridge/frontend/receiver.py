import base64
import io
import json
import logging
import threading
import tempfile
from enum import Enum
from typing import Any, List

import torch
import torch.distributed as dist
import zmq
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wbridge.utils.data import WeightData
from wbridge.utils.distributed import init_custom_process_group

logger = logging.getLogger(__name__)

# Message types for controller <-> receiver communication
METADATA_REQUEST = "metadata_request"
CONNECT_REQUEST = "connect_request"
RECEIVE_REQUEST = "receive_request"
# Scheduler (REQ) -> receiver (REP); replaces the old ready check
UPDATE_REQUEST = "update_request"


class ReceiverState(str, Enum):
    """Receiver lifecycle for controller vs scheduler handoff."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    AWAITING_SCHEDULER_UPDATE = "awaiting_scheduler_update"


class ConnectRequest(BaseModel):
    master_address: str
    master_port: int
    base_rank: int
    world_size: int
    group_name: str
    sender_world_size: int
    backend: str  # "nccl" (gpu) or "gloo" (cpu)


class WeightReceiver:
    def __init__(
        self,
        controller_ipc_name: str,
        rank: int,
        metadata: WeightData,
    ):
        self.controller_ipc_name = controller_ipc_name
        self.scheduler_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.rank = rank
        self.metadata = dict(metadata.meta_dict)
        self._state = ReceiverState.DISCONNECTED
        self.receiver_thread = threading.Thread(
            target=self._receiver_process_entry
        )
        self.receiver_thread.start()

    def stop(self):
        """Stop the receiver process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def request_update(self) -> dict[str, Any]:
        """Scheduler REQ/REP: trigger weight receive when state is AWAITING_SCHEDULER_UPDATE."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 600_000)
        socket.setsockopt(zmq.SNDTIMEO, 600_000)
        try:
            socket.connect(self.scheduler_ipc_name)
            socket.send_string(UPDATE_REQUEST)
            return json.loads(socket.recv_string())
        except Exception as e:
            logger.warning("WeightReceiver request_update failed: %s", e)
            return {"success": False}
        finally:
            socket.close()
            context.term()

    @staticmethod
    def _serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return {
            "state_dict_torch_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }

    def _handle_metadata_request(self, controller_socket: zmq.Socket) -> None:
        response = json.dumps({"rank": self.rank, "metadata": self.metadata})
        controller_socket.send_string(response)

    def _handle_connect_request(
        self, controller_socket: zmq.Socket, data: dict[str, Any]
    ) -> None:
        """
        Handle connect request from controller. This sets up the process group and the device.
        It also destroys the previous process group if it exists.
        """
        if self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE:
            controller_socket.send_string(
                json.dumps(
                    {
                        "status": "error",
                        "detail": "cannot connect while awaiting scheduler update",
                    }
                )
            )
            return
        controller_socket.send_string(json.dumps({"status": "ack"}))
        if getattr(self, "group", None) is not None:
            dist.destroy_process_group(self.group)
        backend = data["backend"]
        self.backend = backend
        self.group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{data['master_address']}:{data['master_port']}",
            world_size=data["world_size"],
            rank=data["rank"],
            group_name=data["group_name"],
        )
        self.device = "cuda" if backend == "nccl" else "cpu"
        self.overlaps: dict[int, WeightData] = {}
        for sender_rank in range(data["sender_world_size"]):
            size_t = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.recv(size_t, src=sender_rank, group=self.group)
            data_t = torch.zeros(
                size_t.item(), dtype=torch.uint8, device=self.device
            )
            dist.recv(data_t, src=sender_rank, group=self.group)
            overlap_dict = json.loads(
                data_t.cpu().numpy().tobytes().decode("utf-8")
            )
            self.overlaps[sender_rank] = WeightData(overlap_dict)
        logger.info(
            "Receiver worker %d joined group %s as rank %d "
            "(world_size=%d, overlaps from %d senders)",
            self.rank, data["group_name"], data["rank"],
            data["world_size"], len(self.overlaps),
        )
        self._state = ReceiverState.CONNECTED

    def _handle_receive_request(self, controller_socket: zmq.Socket) -> None:
        if self._state != ReceiverState.CONNECTED:
            controller_socket.send_string(
                json.dumps(
                    {
                        "status": "error",
                        "detail": "receive requires CONNECTED state",
                    }
                )
            )
            return
        controller_socket.send_string(json.dumps({"status": "ack"}))
        self._state = ReceiverState.AWAITING_SCHEDULER_UPDATE

    def _handle_scheduler_update(
        self, scheduler_socket: zmq.Socket, msg: str
    ) -> None:
        if msg != UPDATE_REQUEST:
            scheduler_socket.send_string(
                json.dumps({"success": False, "message": "unknown message"})
            )
            return
        if self._state != ReceiverState.AWAITING_SCHEDULER_UPDATE:
            scheduler_socket.send_string(json.dumps({"success": False}))
            return
        state_dict = self._receive_weights()
        self._state = ReceiverState.CONNECTED
        payload = self._serialize_state_dict(state_dict)
        scheduler_socket.send_string(
            json.dumps({"success": True, **payload})
        )

    def _receiver_process_entry(
        self
    ):
        """Entry point for the WeightReceiver subprocess. Handles both controller (DEALER)
        and scheduler (REP) requests.
        """
        context = zmq.Context()
        poller = zmq.Poller()

        # DEALER: connect to controller's ROUTER for metadata requests
        controller_socket = context.socket(zmq.DEALER)
        controller_socket.setsockopt_string(zmq.IDENTITY, f"worker-{self.rank}")
        controller_socket.connect(self.controller_ipc_name)
        poller.register(controller_socket, zmq.POLLIN)

        # REP: bind for scheduler's REQ (update -> recv weights)
        scheduler_socket = context.socket(zmq.REP)
        scheduler_socket.bind(self.scheduler_ipc_name)
        poller.register(scheduler_socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll())
            if controller_socket in socks:
                msg = controller_socket.recv_string()
                data = json.loads(msg)
                req_type = data.get("type")
                if req_type == METADATA_REQUEST:
                    self._handle_metadata_request(controller_socket)
                elif req_type == CONNECT_REQUEST:
                    self._handle_connect_request(controller_socket, data)
                elif req_type == RECEIVE_REQUEST:
                    self._handle_receive_request(controller_socket)
                else:
                    raise ValueError(f"Unknown message from controller: {msg}")
            if scheduler_socket in socks:
                msg = scheduler_socket.recv_string()
                self._handle_scheduler_update(scheduler_socket, msg)

    def _receive_weights(self) -> dict[str, torch.Tensor]:
        """irecv overlap bytes from each sender, unpack to tensors (same order as sender ``pack_for``)."""
        ops = []
        buffers: list[tuple[int, torch.Tensor, WeightData]] = []
        device = "cuda" if self.backend == "nccl" else "cpu"
        for sender_rank, overlap in self.overlaps.items():
            if not overlap.meta_dict:
                continue
            nbytes = overlap.total_nbytes()
            tensor = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            logger.info(
                "Receiver %d <- Sender %d: %d bytes",
                self.rank, sender_rank, nbytes,
            )
            buffers.append((sender_rank, tensor, overlap))
            ops.append(
                dist.P2POp(
                    op=torch.distributed.irecv,
                    tensor=tensor,
                    peer=sender_rank,
                    group=self.group,
                )
            )
        dist.batch_isend_irecv(ops)
        for op in ops:
            op.wait()
        merged: dict[str, torch.Tensor] = {}
        for sender_rank, tensor, overlap in buffers:
            partial = overlap.tensors_from_flat(tensor)
            for name, t in partial.items():
                if name in merged:
                    raise ValueError(
                        f"duplicate tensor name {name!r} from sender {sender_rank}"
                    )
                merged[name] = t
        return merged


class WeightReceiverController:
    """
    Server for receiving weights from a WeightSender.

    It is used to pass metadata and dispatching requests to underlying WeightReceiver instances.
    When created, it creates an IPC name file for ROUTER/DEALER communication with receivers.
    """

    def __init__(self, app: FastAPI, worker_num: int = 0):
        # Create IPC name like PortArgs.init_new
        self._ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self._receiver_identities: dict = {}  # identity bytes -> rank

        self._context = zmq.Context()
        self._router_socket = self._context.socket(zmq.ROUTER)
        self._router_socket.bind(self._ipc_name)

        self.router = APIRouter()
        self.router.add_api_route(
            path="/wbridge/metadata", endpoint=self.metadata, methods=["GET"]
        )
        self.router.add_api_route(
            path="/wbridge/connect", endpoint=self.connect, methods=["POST"]
        )
        self.router.add_api_route(
            path="/wbridge/receive", endpoint=self.receive_weights, methods=["POST"]
        )
        app.include_router(self.router)
        
        
    @property
    def ipc_name(self) -> str:
        """The IPC name for receivers to connect (ROUTER binds here)."""
        return self._ipc_name

    def set_worker_num(self, worker_num: int) -> None:
        """Set the expected number of receiver workers. Called after schedulers are launched."""
        self._worker_num = worker_num
        self._receiver_identities = [b"worker-%d" % rank for rank in range(worker_num)]

    def _query_receivers_metadata(self) -> List[dict]:
        """Query all connected receivers via ROUTER/DEALER (``{"type": "metadata_request"}``).
        Collects until worker_num responses received or timeout.
        Each item includes ``rank`` and ``metadata`` (per worker JSON from the receiver).
        """
        metadata_msg = json.dumps({"type": METADATA_REQUEST}).encode("utf-8")
        for identity in self._receiver_identities:
            self._router_socket.send_multipart([identity, metadata_msg])

        results = []
        while len(results) < self._worker_num:
            parts = self._router_socket.recv_multipart()
            assert len(parts) >= 2, "Invalid response from receiver"
            resp = json.loads(parts[1].decode("utf-8"))
            results.append(resp)
        return sorted(results, key=lambda r: r.get("rank", -1))

    async def metadata(self):
        """Return metadata from all receivers (rank + weight shard metadata per receiver)."""
        results = self._query_receivers_metadata()
        return JSONResponse(content=results)

    async def connect(self, request: ConnectRequest):
        """Forward connection info to each worker, wait for acks, then return.

        Each worker is assigned rank ``request.base_rank + worker_index``.
        Workers acknowledge immediately (before blocking on group formation),
        so this endpoint returns as soon as all workers have the info.
        """
        for idx, identity in enumerate(self._receiver_identities):
            connect_msg = json.dumps({
                "type": CONNECT_REQUEST,
                "master_address": request.master_address,
                "master_port": request.master_port,
                "rank": request.base_rank + idx,
                "world_size": request.world_size,
                "group_name": request.group_name,
                "sender_world_size": request.sender_world_size,
                "backend": request.backend,
            })
            self._router_socket.send_multipart(
                [identity, connect_msg.encode("utf-8")]
            )

        acks = 0
        while acks < self._worker_num:
            self._router_socket.recv_multipart()
            acks += 1

        return JSONResponse(content={"status": "success"})

    async def receive_weights(self):
        """Signal workers to enter ``AWAITING_SCHEDULER_UPDATE`` (HTTP ack only). Actual recv runs on scheduler ``update``."""
        for identity in self._receiver_identities:
            receive_msg = json.dumps({"type": RECEIVE_REQUEST})
            self._router_socket.send_multipart(
                [identity, receive_msg.encode("utf-8")]
            )

        acks = 0
        while acks < self._worker_num:
            self._router_socket.recv_multipart()
            acks += 1

        return JSONResponse(content={"status": "success"})
