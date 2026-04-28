"""Local receiver side of Weightbridge: per-worker :class:`WeightReceiver` and HTTP :class:`WeightReceiverController`.

The **controller** exposes FastAPI routes that the training scheduler (or a driver) uses to
connect workers and to signal a weight receive round. The controller forwards JSON control
messages over a ZMQ ROUTER to per-worker DEALER sockets.

Each **WeightReceiver** runs a ZMQ message loop in a daemon thread, updating state and
either staging CPU buffers or, with the main thread, completing NCCL recv and
``load_weights``—see :class:`WeightReceiver` and :meth:`WeightReceiver.request_update`.
"""

import json
import logging
import tempfile
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import zmq
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

from wbridge.backend.router import WBEndpoint
from wbridge.utils.data import ShardSpec

logger = logging.getLogger(__name__)

# Message types for controller <-> receiver communication
CONNECT_REQUEST = "connect_request"
RECEIVE_REQUEST = "receive_request"


class ReceiverState(str, Enum):
    """Receiver lifecycle: ZMQ thread vs. main-thread ``request_update`` coordination."""

    DISCONNECTED = "disconnected"  # No process group; not yet connected
    PENDING_CONNECT = "pending_connect"  # gpu_direct: connect deferred to main thread
    CONNECTED = "connected"  # Ready for receive or idle after a completed round
    AWAITING_SCHEDULER_UPDATE = "awaiting_scheduler_update"  # Staging/recv done; main thread loads


class WeightReceiver(WBEndpoint):
    """Runs a ZMQ message loop in a daemon thread for controller (ROUTER/DEALER) traffic.

    ``gpu_direct`` (NCCL): connect may defer process-group creation to the main thread
    (see :attr:`ReceiverState.PENDING_CONNECT`); :meth:`request_update` runs
    :meth:`_receive_weights` on the main thread after the HTTP receive handshake.

    ``cpu_direct`` (Gloo, CPU wire buffers): the HTTP receive handler acknowledges immediately,
    then a background thread runs :meth:`_receive_weights_staging_only` so trainers can post
    ``isend`` while bytes land in CPU staging buffers. :meth:`request_update` waits for that
    staging to finish, then calls :attr:`load_weights` to copy into GPU ``wksd``.
    """

    def __init__(
        self,
        controller_ipc_name: str,
        rank: int,
        shard_spec: ShardSpec,
        dtype_spec: dict[str, torch.dtype],
        load_weights: Callable[[ShardSpec, dict[str, torch.Tensor]], None],
    ) -> None:
        self.cuda_device = f"cuda:{torch.cuda.current_device()}"

        self.transfer_mode: str | None = None  # set in :meth:`_handle_connect_request`
        self._controller_ipc_name = controller_ipc_name
        self.rank = rank
        self.shard_spec = shard_spec
        self.dtype_spec = dtype_spec
        self.load_weights = load_weights

        self._cpu_recv_buffer: dict[str, torch.Tensor] | None = None

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._ready_event = threading.Event()

        self._state = ReceiverState.DISCONNECTED
        self.device: Optional[str] = None
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


    def request_update(self) -> bool:
        """Apply pending connect, or finish a receive and return ``True`` when weights were loaded."""
        with self._lock:
            if self._state == ReceiverState.PENDING_CONNECT: # Only in gpu_direct transfer mode
                assert self._pending_connect_data is not None
                self._send_ack()
                self.set_up_connection(**self._pending_connect_data)
                assert self.transfer_mode == "gpu_direct", "Delayed connect should be in gpu_direct transfer mode"
                self._pending_connect_data = None
                self._state = ReceiverState.CONNECTED
                return False
            if self._state == ReceiverState.AWAITING_SCHEDULER_UPDATE:
                if self.transfer_mode == "gpu_direct":
                    self._send_ack()
                    self._receive_weights()
                elif self.transfer_mode == "cpu_direct":
                    assert self._cpu_recv_buffer is not None
                    self.load_weights(self.shard_spec, self._cpu_recv_buffer)
                    del self._cpu_recv_buffer
                    self._cpu_recv_buffer = None
                self._state = ReceiverState.CONNECTED
                return True
            return False


    def _send_ack(self) -> None:
        """Send an ack message to the controller."""
        self._controller_socket.send_string(json.dumps({"status": "ack"}))

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

        if data["transfer_mode"] == "gpu_direct":
            self._pending_connect_data = dict(data)
            self._state = ReceiverState.PENDING_CONNECT
            return

        self._send_ack()
        self.set_up_connection(**data)
        self._state = ReceiverState.CONNECTED


    def _handle_receive_request(self, controller_socket: zmq.Socket) -> None:
        """Caller must hold :attr:`_lock` (message loop)."""
        if self._state != ReceiverState.CONNECTED:
            controller_socket.send_string(
                json.dumps({"status": "error", "detail": "requires CONNECTED state"})
            )
            return
        if self.transfer_mode == "cpu_direct":
            self._send_ack()
            self._receive_weights()
        self._state = ReceiverState.AWAITING_SCHEDULER_UPDATE


    def _receive_weights(self) -> None:
        """Multi-round recv: unpack into buffers; call ``load_weights`` when ``apply_load`` is True."""
        router = self.router
        device = self.device
        group = self.group
        assert router is not None and device is not None and group is not None
        for full_spec, overlap_specs in router.local_rounds:
            if not full_spec:
                dist.barrier(group=group)
                continue
            chunks = {
                peer_rank: overlap_spec.make_byte_chunk(self.dtype_spec, device)
                for peer_rank, overlap_spec in overlap_specs.items()
            }
            ops = [
                dist.P2POp(dist.irecv, chunk, peer_rank, group)
                for peer_rank, chunk in chunks.items()
            ]
            dist.barrier(group=group)
            for h in dist.batch_isend_irecv(ops):
                h.wait()

            buf = full_spec.make_named_buffer(self.dtype_spec, device)
            full_spec(buf)[overlap_specs] = chunks
            if self.transfer_mode == "gpu_direct":
                self.load_weights(full_spec, buf)
                del buf
            elif self.transfer_mode == "cpu_direct":
                self._cpu_recv_buffer = buf

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
    """HTTP + ZMQ control surface for a pool of :class:`WeightReceiver` workers.

    Binds a ZMQ ROUTER at :attr:`ipc_name`; one DEALER per worker identity ``worker-<rank>``
    must connect. Routes:

    * ``GET /wbridge/receiver_world`` — world size the scheduler set via :meth:`set_worker_num`
    * ``POST /wbridge/connect`` — forward process-group args to all workers; collect per-worker acks
    * ``POST /wbridge/receive`` — tell workers to prepare for a P2P receive round; acks when ready

    Actual tensor traffic uses the distributed process group created at connect; the scheduler
    is expected to call the worker’s :meth:`WeightReceiver.request_update` after the HTTP
    receive handshake to finish the round (load into model).
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
        """Block until :attr:`_worker_num` JSON replies arrive on the ROUTER (one per worker)."""
        if self._worker_num is None or self._receiver_identities is None:
            raise RuntimeError("ReceiverController.set_worker_num must be called before gather_responses.")
        responses = []
        for _ in range(self._worker_num):
            msg_bytes = self._router_socket.recv_multipart()[1]
            responses.append(json.loads(msg_bytes.decode("utf-8")))
        return responses

    async def get_receiver_world(self):
        """Return the world size of the receiver group."""
        return JSONResponse(content={"status": "success", "world_size": self._worker_num})

    async def connect(self, request: dict[str, Any]):
        """Forward process-group and transfer settings to every worker; success if all ack.

        The same ``request`` dict is sent to every identity; each worker does
        ``data["rank"] += self.rank`` so global ranks are ``request["rank"] + local_index``.

        For ``gpu_direct``, the ZMQ thread does not ack until the main thread runs
        :meth:`WeightReceiver.request_update` in :attr:`ReceiverState.PENDING_CONNECT` (which
        calls :meth:`WeightReceiver.set_up_connection` and then :meth:`_send_ack`). For other
        transfer modes, the thread acks immediately after :meth:`WeightReceiver.set_up_connection`
        in :meth:`WeightReceiver._handle_connect_request`.
        """
        identities = self._receiver_identities
        if identities is None:
            raise RuntimeError("set_worker_num before /wbridge/connect.")

        for identity in identities:
            connect_msg_bytes = json.dumps({"type": CONNECT_REQUEST, **request}).encode("utf-8")
            self._router_socket.send_multipart([identity, connect_msg_bytes])

        success = all(resp["status"] == "ack" for resp in self._gather_responses())
        return JSONResponse(content={"status": "success" if success else "error"})

    async def receive_weights(self):
        """Signal workers to enter ``AWAITING_SCHEDULER_UPDATE`` (HTTP ack only).
        Actual recv runs on scheduler ``update`` call.
        """
        identities = self._receiver_identities
        if identities is None:
            raise RuntimeError("set_worker_num before /wbridge/receive.")

        for identity in identities:
            receive_msg_bytes = json.dumps({"type": RECEIVE_REQUEST}).encode("utf-8")
            self._router_socket.send_multipart([identity, receive_msg_bytes])

        success = all(resp["status"] == "ack" for resp in self._gather_responses())
        return JSONResponse(content={"status": "success" if success else "error"})
