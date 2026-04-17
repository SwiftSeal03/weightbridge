"""
Megatron-Bridge helpers for WeightBridge: build or load a :class:`~wbridge.utils.data.LoadSpec`
that maps HuggingFace checkpoint tensors onto Megatron parameters, and expose a
:class:`~wbridge.backend.sender.WeightSender` for pushing shards to receivers.

The LoadSpec is cached per process rank under ``~/.cache/megatron/loadspec_rank{RANK}.json`` to
avoid repeating expensive inference. Callers that only need inference should still construct
:class:`WBMegatronAdapter` when appropriate; this module does not read ``WBRIDGE_INFER_LOAD_SPEC`` or
``SLIME_WBRIDGE_INFER_LOAD_SPEC`` (conventions for scripts or orchestration layers).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from megatron.bridge import AutoBridge


from wbridge.utils.data import LoadSpec
from wbridge.backend.sender import WeightSender
from wbridge.utils.specgen import infer_load_spec, verify_load_spec
from wbridge.utils.data import shards_numel

logger = logging.getLogger(__name__)

@contextmanager
def patch_megatron_model(model):
    """Temporarily add ``share_embeddings_and_output_weights`` to the unwrapped config if missing.

    ``megatron.bridge`` expects this attribute during conversion; it is removed again after the
    context unless it was already present.
    """
    from megatron.core.utils import unwrap_model

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
        from safetensors.torch import safe_open

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
    from megatron.core.utils import unwrap_model

    inner = unwrap_model(model[0])
    if isinstance(inner, (list, tuple)):
        return list(inner)
    return [inner]


def _wksd_from_conversion_tasks(tasks: list[Any]) -> dict[str, Any]:
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


class WBMegatronAdapter:
    """Tie a loaded Megatron ``model`` to HF weights and a :class:`~wbridge.utils.data.LoadSpec`.

    On construction, builds Megatron-Bridge conversion tasks, loads or infers a LoadSpec (see
    :meth:`_get_load_spec_and_verify`), and creates a :class:`~wbridge.backend.sender.WeightSender`
    with ``*args`` (``world_size``, ``transfer_mode``, ``receiver_urls``, ``master_addr``,
    ``master_port``).
    """

    def __init__(
        self,
        hf_checkpoint: str,
        model: list[torch.nn.Module],
        rank: int,
        *args: Any,
    ) -> None:
        self.hf_checkpoint = hf_checkpoint
        self.model = model
        self.chunks = _megatron_model_chunks(model)
        self.bridge = AutoBridge.from_hf_pretrained(hf_checkpoint)
        with patch_megatron_model(model):
            self.conv_tasks = list(self.bridge.get_conversion_tasks(self.chunks))
        self.wksd = _wksd_from_conversion_tasks(self.conv_tasks)
        self.load_spec_path = Path.home() / ".cache" / "megatron" / f"loadspec_rank{rank}.json"

        self._get_load_spec_and_verify()
        self.sender = WeightSender(rank, *args)
            

    def _get_hf_iter(self) -> Iterator[tuple[str, Any]]:
        """Iterator over HF checkpoint tensors (same tree as ``self.hf_checkpoint``)."""
        return _iter_hf_checkpoint_cpu_tensors(self.hf_checkpoint)

    @torch.inference_mode()
    def _get_load_spec_and_verify(self) -> None:
        """Load or infer ``self.load_spec`` and verify it against HF tensors and Megatron weights.

        Steps:

        1. ``AutoBridge.from_hf_pretrained(self.hf_checkpoint)`` and
           ``get_conversion_tasks(megatron_chunks)`` to obtain CUDA parameter tensors (*wksd*).
        2. If ``self.load_spec_path`` exists, parse it as a :class:`~wbridge.utils.data.LoadSpec` and
           :func:`~wbridge.utils.specgen.verify_load_spec`; return on success.
        3. On any failure (missing file, bad JSON, verification mismatch), log and infer a new spec:
           stream HF tensors, run a ``lw`` closure that mirrors HF→Megatron loading (including
           optional grouped HF params and shared-embedding broadcast), then call
           :func:`~wbridge.utils.specgen.infer_load_spec` and verify again. Persist the result to
           ``self.load_spec_path``.

        Inference can be very slow on large models and reads the HF tree multiple times. In
        distributed runs, keep cache files consistent across ranks (same path visibility and valid
        spec); otherwise ranks can diverge and later collectives may deadlock.

        The decorator applies ``torch.inference_mode()`` because specgen probing uses in-place writes
        on Megatron tensors and must not build autograd graphs.
        """
        # Try to load LoadSpec from cache and verify it
        try:
            with open(self.load_spec_path, encoding="utf-8") as f:
                self.load_spec = LoadSpec(json.load(f))
            verify_load_spec(self._get_hf_iter(), self.wksd, self.load_spec)
            self.shard_spec = self.load_spec.src_spec()
            return
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            
        logger.info("wbridge: Loading LoadSpec from cache failed, will infer a new LoadSpec")

        # Infer a new LoadSpec
        hf_tensor_meta = {k: (tuple(t.shape), t.dtype) for k, t in self._get_hf_iter()}
        
        def lw(weights: Any) -> None:
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
                        if hf_tensor_meta is not None and isinstance(hf_param, dict):
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
                            hf_weights = self.bridge._model_bridge.maybe_modify_loaded_hf_weight(hf_param, hf_chunk)
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

        self.load_spec = infer_load_spec(self._get_hf_iter(), self.wksd, lw)
        self.shard_spec = self.load_spec.src_spec()
        try:
            verify_load_spec(self._get_hf_iter(), self.wksd, self.load_spec)
        except Exception as e:
            logger.error(f"wbridge: Inferred LoadSpec verification failed. This is likely a bug in the LoadSpec inference logic. Please report this to the developers.")
            raise e

        # Save the LoadSpec to cache
        os.makedirs(self.load_spec_path.parent, exist_ok=True)
        with open(self.load_spec_path, "w", encoding="utf-8") as f:
            json.dump(self.load_spec.entries, f, indent=2, sort_keys=True)
        logger.info("wbridge: wrote LoadSpec to %s", self.load_spec_path)
        
        
    def connect(self) -> None:
        self.sender.connect(self.shard_spec)
        self.dtype_spec = self.sender.dtype_spec
        
        
    def send_weights(self) -> None:
        self.sender_buffer = {
            name: torch.empty(shards_numel(self.shard_spec[name]), dtype=self.dtype_spec[name], device="cuda")
            for name in self.shard_spec
        }
        self.load_spec.copy_fromto_sharded(self.shard_spec, self.sender_buffer, self.wksd, src_to_dst=False)
        self.sender.send(self.sender_buffer)
