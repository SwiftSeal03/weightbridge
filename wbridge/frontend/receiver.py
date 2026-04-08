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

from wbridge.utils.data import ShardSpec
from wbridge.utils.distributed import init_custom_process_group

logger = logging.getLogger(__name__)

# Message types for controller <-> receiver communication
CONNECT_REQUEST = "connect_request"
RECEIVE_REQUEST = "receive_request"
# Scheduler (REQ) -> receiver (REP); replaces the old ready check
UPDATE_REQUEST = "update_request"


class ReceiverState(str, Enum):
    """Receiver lifecycle for controller vs scheduler handoff."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    AWAITING_SCHEDULER_UPDATE = "awaiting_scheduler_update"


class WeightReceiver:
    def __init__(
        self,
        controller_ipc_name: str,
        rank: int,
        shard_spec: ShardSpec,
    ):
        self.controller_ipc_name = controller_ipc_name
        self.scheduler_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.rank = rank
        self.shard_spec = shard_spec
        self.state_dict = None
        self._state = ReceiverState.DISCONNECTED
        self.ready_event = threading.Event()
        self.receiver_thread = threading.Thread(
            target=self._receiver_process_entry
        )
        self.receiver_thread.start()
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.scheduler_ipc_name)
        
        if not self.ready_event.wait(timeout=10):
            raise TimeoutError("Receiver thread did not become ready in time")

    def stop(self):
        """Stop the receiver subprocess if ``self.process`` was set (optional hook)."""
        proc = getattr(self, "process", None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join()


    def is_weights_ready(self) -> bool:
        """True after the trainer has POSTed ``/wbridge/receive``; then call :meth:`request_update`."""
        return self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE

    def request_update(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Scheduler REQ/REP: trigger weight receive when state is AWAITING_SCHEDULER_UPDATE."""
        self.state_dict = state_dict # Receive to this state_dict
        self.socket.send_string(UPDATE_REQUEST)
        return json.loads(self.socket.recv_string())["success"]


    def _handle_connect_request(
        self, controller_socket: zmq.Socket, data: dict[str, Any]
    ) -> None:
        """
        Handle connect request from controller. This sets up the process group and the device.
        It also destroys the previous process group if it exists.
        """
        if self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE:
            controller_socket.send_string(
                json.dumps({"status": "error", "detail": "cannot connect while awaiting scheduler update"})
            )
            return
        controller_socket.send_string(json.dumps({"status": "ack"}))
        if getattr(self, "group", None) is not None:
            dist.destroy_process_group(self.group)
            
        # Initialize the process group (pop keys not accepted by init_custom_process_group)
        sender_world_size = int(data.pop("sender_world_size"))
        data["rank"] += self.rank
        is_nccl = data["backend"] == "nccl"
        device_id = torch.device("cuda", torch.cuda.current_device()) if is_nccl else None
        self.group = init_custom_process_group(**data, device_id=device_id)
        self.device = "cuda" if is_nccl else "cpu"
        dist.barrier(group=self.group)
        print(f"barrier passed for rank {self.rank}")
        
        all_specs = [None] * data["world_size"]
        dist.all_gather_object(all_specs, self.shard_spec, group=self.group)
        self.overlaps = {
            rank: overlap for rank, peer_spec in enumerate(all_specs) 
            if rank < sender_world_size and (overlap := ShardSpec.compute_overlap(peer_spec, self.shard_spec))
        }
            
        self._state = ReceiverState.CONNECTED


    def _handle_receive_request(self, controller_socket: zmq.Socket) -> None:
        if self._state != ReceiverState.CONNECTED:
            controller_socket.send_string(
                json.dumps({"status": "error", "detail": "requires CONNECTED state"})
            )
            return
        controller_socket.send_string(json.dumps({"status": "ack"}))
        self._state = ReceiverState.AWAITING_SCHEDULER_UPDATE


    def _handle_scheduler_update(
        self, scheduler_socket: zmq.Socket, msg: str
    ) -> None:
        if self._state != ReceiverState.AWAITING_SCHEDULER_UPDATE:
            scheduler_socket.send_string(
                json.dumps({"success": False, "message": "no pending weights"})
            )
            return
        self._receive_weights()
        self._state = ReceiverState.CONNECTED
        scheduler_socket.send_string(json.dumps({"success": True}))


    def _receiver_process_entry(
        self
    ):
        """Entry point for the WeightReceiver subprocess. Handles both controller (DEALER)
        and scheduler (REP) requests.
        """
        context = zmq.Context()
        poller = zmq.Poller()

        # DEALER: connect to controller's ROUTER for control messages
        controller_socket = context.socket(zmq.DEALER)
        controller_socket.setsockopt_string(zmq.IDENTITY, f"worker-{self.rank}")
        controller_socket.connect(self.controller_ipc_name)
        poller.register(controller_socket, zmq.POLLIN)

        # REP: bind for scheduler's REQ (update -> recv weights)
        scheduler_socket = context.socket(zmq.REP)
        scheduler_socket.bind(self.scheduler_ipc_name)
        poller.register(scheduler_socket, zmq.POLLIN)
        
        self.ready_event.set()

        while True:
            socks = dict(poller.poll())
            if controller_socket in socks:
                msg = controller_socket.recv_string()
                data = json.loads(msg)
                req_type = data.pop("type")
                if req_type == CONNECT_REQUEST:
                    self._handle_connect_request(controller_socket, data)
                elif req_type == RECEIVE_REQUEST:
                    self._handle_receive_request(controller_socket)
                else:
                    raise ValueError(f"Unknown message from controller: {msg}")
            if scheduler_socket in socks:
                msg = scheduler_socket.recv_string()
                if msg == UPDATE_REQUEST:
                    self._handle_scheduler_update(scheduler_socket, msg)
                else:
                    raise ValueError(f"Unknown message from scheduler: {msg}")


    def _receive_weights(self) -> None:
        """Receive overlap bytes from each sender, unpack into ``state_dict`` via :class:`BoundShardSpec`."""
        chunks = {
            sender_rank: torch.zeros(overlap.total_nbytes(), dtype=torch.uint8, device=self.device)
            for sender_rank, overlap in self.overlaps.items()
        }
        ops = [
            dist.P2POp(dist.irecv, chunk, sender_rank, self.group)
            for sender_rank, chunk in chunks.items()
        ]
        for h in dist.batch_isend_irecv(ops):
            h.wait()
        
        self.shard_spec(self.state_dict)[self.overlaps] = chunks


class WeightReceiverController:
    """
    Server for receiving weights from a WeightSender.

    It forwards control messages and dispatches to underlying WeightReceiver instances.
    When created, it creates an IPC name file for ROUTER/DEALER communication with receivers.
    """

    def __init__(self, app: FastAPI):
        # Create IPC name like PortArgs.init_new
        self._ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        self._context = zmq.Context()
        self._router_socket = self._context.socket(zmq.ROUTER)
        self._router_socket.bind(self._ipc_name)

        self.router = APIRouter()
        self.router.add_api_route(
            path="/wbridge/receiver_world", endpoint=self.get_receiver_world, methods=["GET"]
        )
        self.router.add_api_route(
            path="/wbridge/connect", endpoint=self.connect, methods=["POST"]
        )
        self.router.add_api_route(
            path="/wbridge/receive", endpoint=self.receive_weights, methods=["POST"]
        )
        app.include_router(self.router)
        
        # These are set by the scheduler after the receiver workers are launched.
        self._worker_num = None
        self._receiver_identities = None
        
        
    @property
    def ipc_name(self) -> str:
        """The IPC name for receivers to connect (ROUTER binds here)."""
        return self._ipc_name


    def set_worker_num(self, worker_num: int) -> None:
        """Set the expected number of receiver workers. Called after schedulers are launched."""
        self._worker_num = worker_num
        self._receiver_identities = [b"worker-%d" % rank for rank in range(worker_num)]


    def _gather_responses(self) -> List[dict]:
        responses = []
        for _ in range(self._worker_num):
            msg_bytes = self._router_socket.recv_multipart()[1]
            responses.append(json.loads(msg_bytes.decode("utf-8")))
        return responses
        
        
    async def get_receiver_world(self):
        """Return the world size of the receiver group."""
        return JSONResponse(content={"status": "success", "world_size": self._worker_num})


    async def connect(self, request: dict[str, Any]):
        """Forward connection info to each worker, wait for acks, then return.

        Each worker is assigned rank ``request.base_rank + worker_index``.
        Workers acknowledge immediately (before blocking on group formation),
        so this endpoint returns as soon as all workers have the info.
        """
        for idx, identity in enumerate(self._receiver_identities):
            connect_msg_bytes = json.dumps({"type": CONNECT_REQUEST, **request}).encode("utf-8")
            self._router_socket.send_multipart([identity, connect_msg_bytes])

        success = all(resp["status"] == "ack" for resp in self._gather_responses())
        return JSONResponse(content={"status": "success" if success else "error"})


    async def receive_weights(self):
        """Signal workers to enter ``AWAITING_SCHEDULER_UPDATE`` (HTTP ack only). 
        Actual recv runs on scheduler ``update`` call.
        """
        for identity in self._receiver_identities:
            receive_msg_bytes = json.dumps({"type": RECEIVE_REQUEST}).encode("utf-8")
            self._router_socket.send_multipart([identity, receive_msg_bytes])

        success = all(resp["status"] == "ack" for resp in self._gather_responses())
        return JSONResponse(content={"status": "success" if success else "error"})
