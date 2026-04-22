import json
import logging
import tempfile
import threading
from enum import Enum
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import zmq
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

from wbridge.utils.data import ShardSpec, shards_numel
from wbridge.utils.distributed import init_custom_process_group

logger = logging.getLogger(__name__)

# Message types for controller <-> receiver communication
CONNECT_REQUEST = "connect_request"
RECEIVE_REQUEST = "receive_request"


class ReceiverState(str, Enum):
    """Receiver lifecycle for controller vs scheduler handoff."""

    DISCONNECTED = "disconnected"
    PENDING_CONNECT = "pending_connect"
    CONNECTED = "connected"
    AWAITING_SCHEDULER_UPDATE = "awaiting_scheduler_update"


class WeightReceiver:
    """Runs a ZMQ message loop in a daemon thread for controller (ROUTER/DEALER) traffic.

    Gloo (CPU) runs :meth:`_receive_weights` on that worker thread when the HTTP
    receive signal arrives. NCCL defers process-group creation to the main thread
    (see :attr:`ReceiverState.PENDING_CONNECT`) and runs :meth:`_receive_weights`
    on the main thread in :meth:`request_update`.
    """

    def __init__(
        self,
        controller_ipc_name: str,
        rank: int,
        shard_spec: ShardSpec,
        dtype_spec: dict[str, torch.dtype],
    ):
        self._controller_ipc_name = controller_ipc_name
        self.rank = rank
        self.shard_spec = shard_spec
        self.dtype_spec = dtype_spec
        self.cuda_device = f"cuda:{torch.cuda.current_device()}"

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._ready_event = threading.Event()

        self._state = ReceiverState.DISCONNECTED
        self.group = None
        self.device: Optional[str] = None
        self.overlaps: dict = {}
        self.recv_buffer: dict[str, torch.Tensor] = {}
        self._pending_connect_data: Optional[dict[str, Any]] = None

        self._zmq_context = zmq.Context()
        
        self._controller_socket = self._zmq_context.socket(zmq.DEALER)
        self._controller_socket.setsockopt_string(zmq.IDENTITY, f"worker-{self.rank}")
        self._controller_socket.connect(self._controller_ipc_name)
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=60):
            raise TimeoutError("Receiver thread did not become ready in time")


    def stop(self) -> None:
        """Signal the receiver thread to exit and release ZMQ resources."""
        self._shutdown.set()
        self._thread.join(timeout=10.0)
        self._zmq_context.term()
    

    def request_update(self) -> Optional[dict[str, torch.Tensor]]:
        """Apply a pending NCCL connect or hand off received weights when ready; else ``None``."""
        with self._lock:
            if self._state == ReceiverState.PENDING_CONNECT:
                assert self._pending_connect_data is not None
                assert "cuda" in self.device, "Delayed connect should be on CUDA"
                data = dict(self._pending_connect_data)
                self._pending_connect_data = None
                self._send_ack()
                self._install_process_group_and_gather_specs(data)
                self._state = ReceiverState.CONNECTED
                return None
            if self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE:
                if self.device != "cpu":
                    self._send_ack()
                    self._receive_weights()
                self._state = ReceiverState.CONNECTED
                return self.recv_buffer
            return None
        
    
    def _send_ack(self) -> None:
        """Send an ack message to the controller."""
        self._controller_socket.send_string(json.dumps({"status": "ack"}))
        

    def _install_process_group_and_gather_specs(self, data: dict[str, Any]) -> None:
        """Build the process group and gather shard/dtype metadata (shared by connect paths)."""
        sender_world_size = int(data.pop("sender_world_size"))
        logger.error(f"Receiver: {self.rank} data: {data} device: {self.device} real device: {torch.cuda.current_device()}")
        self.group = init_custom_process_group(**data)

        all_shard_specs = [None] * data["world_size"]
        dist.all_gather_object(all_shard_specs, self.shard_spec, group=self.group)
        self.overlaps = {
            rank: overlap
            for rank, spec in enumerate(all_shard_specs)
            if rank < sender_world_size and (overlap := ShardSpec.compute_overlap(self.shard_spec, spec))
        }

        all_dtype_specs = [None] * data["world_size"]
        dist.all_gather_object(all_dtype_specs, self.dtype_spec, group=self.group)

        for dtype_spec in all_dtype_specs:
            if dtype_spec is not None:
                for name, dtype in dtype_spec.items():
                    if name in self.dtype_spec:
                        assert self.dtype_spec[name] == dtype, (
                            f"Dtype mismatch for {name} on rollout rank {data['rank']}"
                        )
                    else:
                        self.dtype_spec[name] = dtype
        
        logger.error(f"Receiver: {self.rank} group initialized on device {self.device}")


    def _handle_connect_request(self, controller_socket: zmq.Socket, data: dict[str, Any]) -> None:
        """
        Handle connect request from controller. This sets up the process group and the device,
        or defers NCCL group creation to the main thread.

        Caller must hold :attr:`_lock` (message loop).
        """
        logger.error(f"Receiver: {self.rank} connect request received")
        if self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE:
            controller_socket.send_string(
                json.dumps({"status": "error", "detail": "cannot connect while awaiting scheduler update"})
            )
            return

        data["rank"] += self.rank
        
        if data["backend"] == "nccl":
            self.device = self.cuda_device
            self._pending_connect_data = dict(data)
            self._state = ReceiverState.PENDING_CONNECT
            logger.error(f"Receiver: {self.rank} NCCL connect deferred to main thread (PENDING_CONNECT)")
            return

        self._send_ack()
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None
        self.device = "cpu"
        self._install_process_group_and_gather_specs(data)
        self._state = ReceiverState.CONNECTED


    def _handle_receive_request(self, controller_socket: zmq.Socket) -> None:
        """Caller must hold :attr:`_lock` (message loop)."""
        if self._state != ReceiverState.CONNECTED:
            controller_socket.send_string(
                json.dumps({"status": "error", "detail": "requires CONNECTED state"})
            )
            return
        if self.device == "cpu":
            self._send_ack()
            self._receive_weights()
        self._state = ReceiverState.AWAITING_SCHEDULER_UPDATE


    def _receive_weights(self) -> None:
        """Receive overlap bytes from each sender, unpack into ``recv_buffer`` via :class:`BoundShardSpec`."""
        chunks = {
            sender_rank: torch.zeros(overlap.nbytes(self.dtype_spec), dtype=torch.uint8, device=self.device)
            for sender_rank, overlap in self.overlaps.items()
        }
        logger.error(f"Receiver: {self.rank} chunks: {[(rank, chunk.size()) for rank, chunk in chunks.items()]}")
        ops = [
            dist.P2POp(dist.irecv, chunk, sender_rank, self.group)
            for sender_rank, chunk in chunks.items()
        ]
        dist.barrier(group=self.group)
        for h in dist.batch_isend_irecv(ops):
            h.wait()
            
        self.recv_buffer = {
            name: torch.empty(shards_numel(self.shard_spec[name]), dtype=self.dtype_spec[name], device=self.device)
            for name, _ in self.shard_spec
        }
        self.shard_spec(self.recv_buffer)[self.overlaps] = chunks


    def _run_loop(self) -> None:
        """Thread entry: DEALER to controller only; signals ready; polls until shutdown."""
        poller = zmq.Poller()
        poller.register(self._controller_socket, zmq.POLLIN)

        self._ready_event.set()

        try:
            while not self._shutdown.is_set():
                socks = dict(poller.poll(500))
                if self._controller_socket in socks:
                    with self._lock:
                        msg = self._controller_socket.recv_string()
                        data = json.loads(msg)
                        req_type = data.pop("type")
                        if req_type == CONNECT_REQUEST:
                            self._handle_connect_request(self._controller_socket, data)
                        elif req_type == RECEIVE_REQUEST:
                            self._handle_receive_request(self._controller_socket)
                        else:
                            raise ValueError(f"Unknown message from controller: {msg}")
        finally:
            self._controller_socket.close(linger=0)


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
