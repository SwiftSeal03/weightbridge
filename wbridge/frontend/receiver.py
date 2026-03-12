import json
import logging
import threading
import tempfile
from typing import List, Optional

import zmq
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Message types for controller <-> receiver communication
METADATA_REQUEST = "metadata_request"
READY_REQUEST = "ready_request"


def _receiver_process_entry(
    controller_ipc_name: str,
    scheduler_ipc_name: str,
    rank: int,
):
    """Entry point for the WeightReceiver subprocess. Handles both controller (DEALER)
    and scheduler (REP) requests.
    """
    
    context = zmq.Context()
    poller = zmq.Poller()
    
    # DEALER: connect to controller's ROUTER for metadata requests
    controller_socket = context.socket(zmq.DEALER)
    controller_socket.setsockopt_string(zmq.IDENTITY, f"worker-{rank}")
    controller_socket.connect(controller_ipc_name)
    poller.register(controller_socket, zmq.POLLIN)

    # REP: bind for scheduler's REQ (ready check)
    scheduler_socket = context.socket(zmq.REP)
    scheduler_socket.bind(scheduler_ipc_name)
    poller.register(scheduler_socket, zmq.POLLIN)
    
    while True:
        socks = dict(poller.poll())
        if controller_socket in socks:
            # Metadata request from controller
            msg = controller_socket.recv_string()
            if msg == METADATA_REQUEST:
                response = json.dumps({"rank": rank})
                controller_socket.send_string(response)
        if scheduler_socket in socks:
            # Ready check from scheduler
            msg = scheduler_socket.recv_string()
            if msg == READY_REQUEST:
                # For now just return False
                scheduler_socket.send_string(json.dumps({"ready": False}))


class WeightReceiver:
    def __init__(
        self,
        controller_ipc_name: str,
        rank: int,
    ):
        self.controller_ipc_name = controller_ipc_name
        self.scheduler_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.rank = rank
        self.receiver_thread = threading.Thread(
            target=_receiver_process_entry,
            args=(self.controller_ipc_name, self.scheduler_ipc_name, self.rank),
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


class WeightReceiverController:
    """
    Server for receiving weights from a WeightSender.

    It is used to pass metadata and dispatching requests to underlying WeightReceiver instances.
    When created, it creates an IPC name file for ROUTER/DEALER communication with receivers.
    """

    def __init__(self, app: FastAPI, worker_num: int = 0):
        # Create IPC name like PortArgs.init_new
        self._ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self._worker_num = worker_num
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

    def _query_receivers_metadata(self) -> List[int]:
        """Query all connected receivers via ROUTER/DEALER, sort by rank.
        Collects until worker_num responses received or timeout.
        Returns list of ranks.
        """
        
        worker_identities = [b"worker-%d" % rank for rank in range(self._worker_num)]
        # Send METADATA_REQUEST to each known identity
        for identity in worker_identities:
            self._router_socket.send_multipart([identity, METADATA_REQUEST.encode(encoding="utf-8")])

        # Collect responses until we have worker_num or timeout
        results = []
        while len(results) < self._worker_num:
            parts = self._router_socket.recv_multipart()
            assert len(parts) >= 2, "Invalid response from receiver"
            resp = json.loads(parts[1].decode("utf-8"))
            results.append(resp)
        return [r.get("rank") for r in results]

    async def metadata(self):
        ranks = self._query_receivers_metadata()
        return JSONResponse(content=ranks)

    async def connect(self):
        return JSONResponse(content={"status": "success"})

    async def receive_weights(self):
        return JSONResponse(content={"status": "success"})
