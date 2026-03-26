import json
import logging
import threading
import tempfile
from typing import List, Optional

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
READY_REQUEST = "ready_request"
CONNECT_REQUEST = "connect_request"


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
        self.metadata = metadata.to_metadata_dict()
        self.receiver_thread = threading.Thread(
            target=self._receiver_process_entry
        )
        self.receiver_thread.start()

    def stop(self):
        """Stop the receiver process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def is_weights_ready(self) -> bool:
        """Check if new weights are ready via REQ/REP. For now returns False."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            socket.connect(self.scheduler_ipc_name)
            socket.send_string(READY_REQUEST)
            response = json.loads(socket.recv_string())
            return response.get("ready", False)
        except Exception as e:
            logger.warning("WeightReceiver ready check failed: %s", e)
            return False
        finally:
            socket.close()
            context.term()
            
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

        # REP: bind for scheduler's REQ (ready check)
        scheduler_socket = context.socket(zmq.REP)
        scheduler_socket.bind(self.scheduler_ipc_name)
        poller.register(scheduler_socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll())
            if controller_socket in socks:
                msg = controller_socket.recv_string()
                if msg == METADATA_REQUEST:
                    response = json.dumps({"rank": self.rank, "metadata": self.metadata})
                    controller_socket.send_string(response)
                else:
                    data = json.loads(msg)
                    if data.get("type") == CONNECT_REQUEST:
                        # Acknowledge before blocking on group formation
                        controller_socket.send_string(json.dumps({"status": "ack"}))
                        backend = data["backend"]
                        self.group = init_custom_process_group(
                            backend=backend,
                            init_method=f"tcp://{data['master_address']}:{data['master_port']}",
                            world_size=data["world_size"],
                            rank=data["rank"],
                            group_name=data["group_name"],
                        )
                        # Receive overlap metadata from each sender
                        device = "cuda" if backend == "nccl" else "cpu"
                        self.overlaps: dict[int, WeightData] = {}
                        for sender_rank in range(data["sender_world_size"]):
                            size_t = torch.zeros(1, dtype=torch.long, device=device)
                            dist.recv(size_t, src=sender_rank, group=self.group)
                            data_t = torch.zeros(
                                size_t.item(), dtype=torch.uint8, device=device
                            )
                            dist.recv(data_t, src=sender_rank, group=self.group)
                            overlap_dict = json.loads(
                                data_t.cpu().numpy().tobytes().decode("utf-8")
                            )
                            self.overlaps[sender_rank] = WeightData.from_metadata_dict(
                                overlap_dict
                            )
                        logger.info(
                            "Receiver worker %d joined group %s as rank %d "
                            "(world_size=%d, overlaps from %d senders)",
                            self.rank, data["group_name"], data["rank"],
                            data["world_size"], len(self.overlaps),
                        )
                        logger.info(
                            "overlaps: %s", json.dumps([d.to_metadata_dict() for d in self.overlaps.values()], indent=4),
                        )
            if scheduler_socket in socks:
                # Ready check from scheduler
                msg = scheduler_socket.recv_string()
                if msg == READY_REQUEST:
                    # For now just return False
                    scheduler_socket.send_string(json.dumps({"ready": False}))


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
        """Query all connected receivers via ROUTER/DEALER.
        Collects until worker_num responses received or timeout.
        Returns list of {rank, metadata} from each receiver.
        """
        for identity in self._receiver_identities:
            self._router_socket.send_multipart([identity, METADATA_REQUEST.encode(encoding="utf-8")])

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
        return JSONResponse(content={"status": "success"})
