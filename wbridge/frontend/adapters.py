"""Shared WeightBridge frontend adapters.

These adapters hold the common :class:`~wbridge.utils.data.LoadSpec` lifecycle used by every
framework-specific frontend: load a cached spec from disk, or infer one by streaming HF tensors
through a framework ``load_weights`` callable into a GPU ``wksd`` (weight state dict). Subclasses
(``WBMegatronAdapter``, ``WBSGLangAdapter``) only need to build an :class:`AdapterContext` and
forward it to :class:`BaseAdapter`.

:class:`SenderAdapter` wraps a :class:`~wbridge.backend.sender.WeightSender`; :class:`ReceiverAdapter`
wraps a :class:`~wbridge.backend.receiver.WeightReceiver` and copies received buffers back into
``wksd`` in :meth:`ReceiverAdapter.try_receive_weights`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from wbridge.backend.receiver import WeightReceiver
from wbridge.backend.sender import SenderArgs, WeightSender
from wbridge.utils.data import LoadSpec, ShardSpec, shards_numel
from wbridge.utils.specgen import infer_load_spec, verify_load_spec

logger = logging.getLogger(__name__)

WeightsIterFactory = Callable[[], Iterator[tuple[str, torch.Tensor]]]
LoadWeightsFn = Callable[[Iterable[tuple[str, torch.Tensor]]], None]


@dataclass
class AdapterContext:
    """Framework-specific inputs every adapter needs to build / verify a LoadSpec.

    Attributes:
        hf_iter_factory: Zero-arg callable returning a fresh ``(name, cpu_tensor)`` iterator over
            the HF checkpoint. Called multiple times during inference / verification.
        wksd: GPU "worker state dict" mapping framework parameter names to the tensors that should
            ultimately receive the loaded weights.
        load_weights: Framework callable with the same contract as ``model.load_weights`` --
            consume an HF ``(name, tensor)`` iterator and write into ``wksd``. Used as a probe by
            :func:`~wbridge.utils.specgen.infer_load_spec`.
        load_spec_path: Per-rank JSON cache path for the inferred LoadSpec.
        rank: Adapter rank, also used in the cache filename suffix during atomic writes.
    """

    hf_iter_factory: WeightsIterFactory
    wksd: dict[str, torch.Tensor]
    load_weights: LoadWeightsFn
    load_spec_path: str | Path
    rank: int


def _dtype_spec_from_load_spec(
    load_spec: LoadSpec, wksd: dict[str, torch.Tensor]
) -> dict[str, torch.dtype]:
    """Per-HF-name dtypes for :class:`~wbridge.backend.receiver.WeightReceiver`.

    For a source name mapped to multiple destinations, pick the widest dtype so a single buffer
    can safely hold every destination's view.
    """
    return {
        hf_name: max((wksd[wk_name].dtype for wk_name in entry), key=lambda d: d.itemsize)
        for hf_name, entry in load_spec.entries.items()
    }


class BaseAdapter:
    """Shared :class:`~wbridge.utils.data.LoadSpec` lifecycle.

    Runs :meth:`ensure_load_spec` from :meth:`__init__`: parse the cached spec at
    ``ctx.load_spec_path`` (validated with :func:`~wbridge.utils.specgen.verify_load_spec`) or
    infer a new one with :func:`~wbridge.utils.specgen.infer_load_spec` and persist it atomically.
    The resulting :class:`~wbridge.utils.data.LoadSpec`, per-HF-name dtypes (``dtype_spec``), and
    HF wire layout (``src_shard_spec``) are stored on ``self``.
    """

    def __init__(self, ctx: AdapterContext) -> None:
        self.ctx = ctx
        self.wksd = ctx.wksd
        self.load_spec_path = Path(ctx.load_spec_path)

        self.load_spec: LoadSpec
        self.dtype_spec: dict[str, torch.dtype]
        self.src_shard_spec: ShardSpec
        self.ensure_load_spec()

    def ensure_load_spec(self) -> None:
        ctx = self.ctx
        loaded = False
        if self.load_spec_path.exists():
            try:
                with open(self.load_spec_path, encoding="utf-8") as f:
                    self.load_spec = LoadSpec.from_jsonable(json.load(f))
                verify_load_spec(ctx.hf_iter_factory(), ctx.wksd, self.load_spec)
                loaded = True
            except Exception as e:
                try:
                    self.load_spec_path.unlink()
                except OSError:
                    pass
                logger.info(
                    "wbridge adapter rank %s: cached LoadSpec invalid (%s); removed file and inferring",
                    ctx.rank,
                    e,
                )

        if not loaded:
            self.load_spec = infer_load_spec(ctx.hf_iter_factory(), ctx.wksd, ctx.load_weights)
            verify_load_spec(ctx.hf_iter_factory(), ctx.wksd, self.load_spec)
            self._persist_load_spec()

        self.dtype_spec = _dtype_spec_from_load_spec(self.load_spec, ctx.wksd)
        self.src_shard_spec = self.load_spec.src_spec()

    def _persist_load_spec(self) -> None:
        self.load_spec_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.load_spec_path.with_suffix(f".tmp.{self.ctx.rank}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.load_spec.entries, f, indent=2, sort_keys=True)
        tmp.replace(self.load_spec_path)
        logger.info("wbridge adapter rank %s: wrote LoadSpec to %s", self.ctx.rank, self.load_spec_path)


class SenderAdapter(BaseAdapter):
    """Frontend-side sender: owns a :class:`~wbridge.backend.sender.WeightSender`.

    The :class:`~wbridge.backend.sender.WeightSender` is constructed in :meth:`__init__` from the
    transport args. Call :meth:`connect` once to join the sender process group, then
    :meth:`send_weights` per weight update.
    """

    def __init__(self, ctx: AdapterContext, args: SenderArgs) -> None:
        super().__init__(ctx)
        self.sender = WeightSender(ctx.rank, args)

    def connect(self) -> None:
        self.sender.connect(self.src_shard_spec)

    def send_weights(self) -> None:
        buf = {
            name: torch.empty(
                shards_numel(self.src_shard_spec[name]),
                dtype=self.dtype_spec[name],
                device=self.sender.device,
            )
            for name, _ in self.src_shard_spec
        }
        self.load_spec.copy_fromto_sharded(self.src_shard_spec, buf, self.wksd, src_to_dst=False)
        self.sender.send(buf)


class ReceiverAdapter(BaseAdapter):
    """Frontend-side receiver: owns a :class:`~wbridge.backend.receiver.WeightReceiver`.

    The receiver is created eagerly in :meth:`__init__` so it can handshake with the controller
    before any transfer begins.
    """

    def __init__(self, ctx: AdapterContext, controller_ipc_name: str) -> None:
        super().__init__(ctx)
        self.controller_ipc_name = controller_ipc_name
        self.receiver = WeightReceiver(
            controller_ipc_name,
            ctx.rank,
            self.src_shard_spec,
            self.dtype_spec,
        )

    def try_receive_weights(self) -> bool:
        """Apply a pending weight update into ``ctx.wksd`` if one is ready.

        Returns ``True`` if an update was consumed, ``False`` if nothing was ready.
        """
        buf = self.receiver.request_update()
        if buf is None:
            return False
        self.load_spec.copy_fromto_sharded(
            self.src_shard_spec,
            buf,
            self.wksd,
            src_to_dst=True,
        )
        return True
