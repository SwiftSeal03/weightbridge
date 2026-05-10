"""Megatron-Bridge Trainer Worker frontend for WeightBridge.

Builds the HF tensor iterator, CUDA ``wksd`` (mapped from Megatron-Bridge conversion tasks), and
``load_weights`` closure that mirrors HF \u2192 Megatron loading, then reuses
:class:`~wbridge.frontend.adapters.SenderAdapter` for the LoadSpec lifecycle and the
:class:`~wbridge.backend.sender.WeightSender` plumbing.

The LoadSpec is cached under ``~/.cache/megatron/loadspec_rank{RANK}.json`` to avoid repeating
expensive inference across runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from megatron.bridge import AutoBridge

from wbridge.backend.sender import SenderArgs
from wbridge.frontend.adapters import AdapterContext, SenderAdapter


@contextmanager
def patch_megatron_model(model):
    """Temporarily add ``share_embeddings_and_output_weights`` to the unwrapped config if missing.

    ``megatron.bridge`` expects this attribute during conversion; it is removed again after the
    context unless it was already present.
    """
    from megatron.core.utils import unwrap_model  # pyright: ignore[reportMissingImports]

    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def _iter_hf_checkpoint_cpu_tensors(hf_path: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, tensor)`` from a HuggingFace-style directory on CPU.

    Prefers ``*.safetensors`` shards; otherwise accepts a single ``pytorch_model*.bin`` state dict.
    """
    root = Path(hf_path)
    st_files = sorted(root.glob("*.safetensors"))
    if st_files:
        from safetensors import safe_open

        for fp in st_files:
            with safe_open(str(fp), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    yield k, sf.get_tensor(k).contiguous()
        return

    bins = sorted(root.glob("pytorch_model*.bin"))
    if len(bins) == 1:
        try:
            blob = torch.load(bins[0], map_location="cpu", weights_only=True)
        except TypeError:
            blob = torch.load(bins[0], map_location="cpu")
        if not isinstance(blob, dict):
            raise TypeError(f"Expected state_dict in {bins[0]}")
        for k, v in blob.items():
            if torch.is_tensor(v):
                yield k, v.contiguous()
        return

    raise FileNotFoundError(
        f"No *.safetensors or single pytorch_model*.bin under {hf_path!r} "
        "(needed for LoadSpec inference)."
    )


def _megatron_model_chunks(model: list[Any]) -> list[Any]:
    """Return the logical model chunk(s) Megatron-Bridge expects from a slime ``model`` list."""
    from megatron.core.utils import unwrap_model  # pyright: ignore[reportMissingImports]

    inner = unwrap_model(model[0])
    if isinstance(inner, (list, tuple)):
        return list(inner)
    return [inner]


def _wksd_from_conversion_tasks(tasks: list[Any]) -> dict[str, torch.Tensor]:
    """Map Megatron parameter names to CUDA tensors referenced by Bridge conversion tasks."""
    wksd: dict[str, torch.Tensor] = {}
    for task in tasks:
        if task is None:
            continue
        pw = getattr(task, "param_weight", None)
        if pw is None or not pw.is_cuda:
            continue
        key = getattr(task, "param_name", None)
        if not key:
            continue
        wksd[key] = pw.detach()
    return wksd


class WBMegatronAdapter(SenderAdapter):
    """Tie a loaded Megatron Trainer Worker ``model`` to HF weights and send shards via
    :class:`~wbridge.frontend.adapters.SenderAdapter`.

    ``connect()`` and ``send_weights()`` are inherited directly from
    :class:`~wbridge.frontend.adapters.SenderAdapter`; ``sender_args`` is forwarded opaquely to
    :class:`~wbridge.backend.sender.WeightSender`, which is the only place the dataclass is
    unpacked.
    """

    def __init__(
        self,
        hf_checkpoint: str,
        model: list[torch.nn.Module],
        rank: int,
        sender_args: SenderArgs,
    ) -> None:
        self.hf_checkpoint = hf_checkpoint
        self.model = model
        self.chunks = _megatron_model_chunks(model)
        self.bridge = AutoBridge.from_hf_pretrained(hf_checkpoint)
        with patch_megatron_model(model):
            self.conv_tasks = list(self.bridge.get_conversion_tasks(self.chunks))
        wksd = _wksd_from_conversion_tasks(self.conv_tasks)

        self._hf_tensor_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] | None = None

        ctx = AdapterContext(
            hf_iter_factory=self._get_hf_iter,
            wksd=wksd,
            load_weights=self._load_weights,
            load_spec_path=Path.home() / ".cache" / "megatron" / f"loadspec_rank{rank}.json",
            rank=rank,
        )
        super().__init__(ctx, sender_args)

    def _get_hf_iter(self) -> Iterator[tuple[str, Any]]:
        """Fresh iterator over HF checkpoint tensors."""
        return _iter_hf_checkpoint_cpu_tensors(self.hf_checkpoint)

    @torch.inference_mode()
    def _load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Mirror HF \u2192 Megatron loading (including grouped HF params and shared-embedding broadcast).

        ``torch.inference_mode`` is required because specgen probing does in-place writes on
        Megatron tensors and must not build autograd graphs.
        """
        if self._hf_tensor_meta is None:
            self._hf_tensor_meta = {
                k: (tuple(t.shape), t.dtype) for k, t in self._get_hf_iter()
            }
        hf_tensor_meta = self._hf_tensor_meta

        hf_chunk = dict(weights)
        hf_cache: dict[str, Any] = {}

        with patch_megatron_model(self.model):
            for task in self.conv_tasks:
                if task is None or getattr(task, "megatron_module", None) is None:
                    continue
                pw = getattr(task, "param_weight", None)
                if pw is None:
                    continue
                try:
                    hf_param = task.mapping.hf_param
                    is_grouped = getattr(task.mapping, "is_grouped_export", False)
                    hf_param_key = str(hf_param)
                    if isinstance(hf_param, dict):
                        if all(v not in hf_chunk for v in hf_param.values()):
                            continue
                        for _logical, ckpt_name in hf_param.items():
                            if ckpt_name in hf_chunk:
                                continue
                            if ckpt_name not in hf_tensor_meta:
                                continue
                            sh, dt = hf_tensor_meta[ckpt_name]
                            hf_chunk[ckpt_name] = torch.zeros(sh, dtype=dt, device="cpu")
                    if is_grouped and hf_param_key in hf_cache:
                        hf_weights = hf_cache[hf_param_key]
                    else:
                        hf_weights = self.bridge._model_bridge.maybe_modify_loaded_hf_weight(
                            hf_param, hf_chunk
                        )
                    if is_grouped:
                        hf_cache[hf_param_key] = hf_weights
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
                if hf_weights is None:
                    continue
                converted = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
                pw.copy_(converted)
            bcast = getattr(self.bridge._model_bridge, "_broadcast_shared_embeddings", None)
            if bcast is not None:
                bcast(self.chunks)
