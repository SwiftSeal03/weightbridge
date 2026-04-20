"""
Minimal WeightBridge adapters for examples: infer :class:`~wbridge.utils.data.LoadSpec` from an
HF tensor iterator, GPU ``wksd``, and a ``load_weights`` callable (same contract as typical
``load_weights`` hooks), then drive :class:`~wbridge.backend.sender.WeightSender` or
:class:`~wbridge.backend.receiver.WeightReceiver`.

*load_weights* maps a full HF dict (from the iterator) into *wksd*; call it to populate *wksd*
before :meth:`ensure_load_spec` / :func:`~wbridge.utils.specgen.verify_load_spec`. HF shard layout
on the wire comes from :meth:`~wbridge.utils.data.LoadSpec.src_spec` after inference.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch

from wbridge.backend.receiver import WeightReceiver
from wbridge.backend.sender import WeightSender
from wbridge.utils.data import LoadSpec, ShardSpec, shards_numel
from wbridge.utils.specgen import infer_load_spec, verify_load_spec

logger = logging.getLogger(__name__)

WeightsIterFactory = Callable[[], Iterator[tuple[str, torch.Tensor]]]
LoadWeightsFn = Callable[[Iterable[tuple[str, torch.Tensor]]], None]


def dtype_spec_from_load_spec(load_spec: LoadSpec, wksd: dict[str, torch.Tensor]) -> dict[str, torch.dtype]:
    """Per-HF-name dtypes for :class:`~wbridge.backend.receiver.WeightReceiver`."""
    return {
        hf_name: max((wksd[wk_name].dtype for wk_name in entry), key=lambda d: d.itemsize)
        for hf_name, entry in load_spec.entries.items()
    }


class _ExampleAdapterBase:
    """Shared LoadSpec path: try :meth:`~wbridge.utils.data.LoadSpec.from_jsonable` on disk, else infer."""

    def __init__(
        self,
        hf_iter_factory: WeightsIterFactory,
        wksd: dict[str, torch.Tensor],
        load_weights: LoadWeightsFn,
        load_spec_path: str | Path,
        rank: int,
    ) -> None:
        self.hf_iter_factory = hf_iter_factory
        self.wksd = wksd
        self.load_weights = load_weights
        self.load_spec_path = Path(load_spec_path)
        self.rank = rank
        self.load_spec: LoadSpec | None = None
        self.dtype_spec: dict[str, torch.dtype] | None = None
        self.src_shard_spec: ShardSpec | None = None

    def ensure_load_spec(self) -> None:
        if self.load_spec is not None:
            return
        loaded = False
        if self.load_spec_path.exists():
            try:
                with open(self.load_spec_path, encoding="utf-8") as f:
                    self.load_spec = LoadSpec.from_jsonable(json.load(f))
                verify_load_spec(self.hf_iter_factory(), self.wksd, self.load_spec)
                loaded = True
            except Exception as e:
                self.load_spec = None
                try:
                    self.load_spec_path.unlink()
                except OSError:
                    pass
                logger.info(
                    "example adapter rank %s: cached LoadSpec invalid (%s); removed file and inferring",
                    self.rank,
                    e,
                )
        if not loaded:
            self.load_spec = infer_load_spec(self.hf_iter_factory(), self.wksd, self.load_weights)
            verify_load_spec(self.hf_iter_factory(), self.wksd, self.load_spec)
            self._persist_load_spec()

        self.dtype_spec = dtype_spec_from_load_spec(self.load_spec, self.wksd)
        self.src_shard_spec = self.load_spec.src_spec()

    def _persist_load_spec(self) -> None:
        self.load_spec_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.load_spec_path.with_suffix(f".tmp.{self.rank}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.load_spec.entries, f, indent=2, sort_keys=True)
        tmp.replace(self.load_spec_path)


class ExampleSenderAdapter(_ExampleAdapterBase):
    """Infer (or load) LoadSpec, then connect a :class:`~wbridge.backend.sender.WeightSender` and send."""

    def __init__(
        self,
        hf_iter_factory: WeightsIterFactory,
        wksd: dict[str, torch.Tensor],
        load_weights: LoadWeightsFn,
        load_spec_path: str | Path,
        rank: int,
    ) -> None:
        super().__init__(hf_iter_factory, wksd, load_weights, load_spec_path, rank)
        self._sender: WeightSender | None = None

    def connect(
        self,
        *,
        transfer_mode: str,
        receiver_urls: list[str],
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self.ensure_load_spec()
        assert self.src_shard_spec is not None
        self._sender = WeightSender(
            self.rank,
            world_size,
            transfer_mode,
            receiver_urls,
            master_addr,
            master_port,
        )
        self._sender.connect(self.src_shard_spec)

    def send(self) -> None:
        if self._sender is None:
            raise RuntimeError("ExampleSenderAdapter.send: call connect() first")
        assert self.load_spec is not None and self.src_shard_spec is not None and self.dtype_spec is not None
        buf = {
            name: torch.empty(
                shards_numel(self.src_shard_spec[name]),
                dtype=self.dtype_spec[name],
                device=self._sender.device
            )
            for name, _ in self.src_shard_spec
        }
        self.load_spec.copy_fromto_sharded(self.src_shard_spec, buf, self.wksd, src_to_dst=False)
        self._sender.send(buf)


class ExampleReceiverAdapter(_ExampleAdapterBase):
    """Load or infer LoadSpec like the sender, then own a :class:`~wbridge.backend.receiver.WeightReceiver`."""

    def make_receiver(self, controller_ipc_name: str) -> WeightReceiver:
        self.ensure_load_spec()
        assert self.dtype_spec is not None and self.src_shard_spec is not None
        return WeightReceiver(
            controller_ipc_name,
            self.rank,
            self.src_shard_spec,
            self.dtype_spec,
        )

    def apply_recv_buffer(self, recv_buffer: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
        assert self.load_spec is not None and self.src_shard_spec is not None
        self.load_spec.copy_fromto_sharded(
            self.src_shard_spec,
            recv_buffer,
            target,
            src_to_dst=True,
        )
