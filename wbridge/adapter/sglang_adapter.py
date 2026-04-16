"""
SGLang to WeightBridge format conversion.
"""

from __future__ import annotations


import torch
import os
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from sglang.srt.model_executor.model_runner import ModelRunner

from wbridge.utils.data import LoadSpec
from wbridge.utils.specgen import infer_load_spec, verify_load_spec
from wbridge.frontend.receiver import WeightReceiver

logger = logging.getLogger(__name__)

class WBSGLangAdapter:
    def __init__(
        self,
        model_runner: ModelRunner,
        rank: int,
        controller_ipc_name: str
    ):
        self.model = model_runner.model
        self.model_config = model_runner.model_config
        self.loader = model_runner.loader
        self.load_spec_path = Path.home() / ".cache" / "sglang" / f"loadspec_rank{rank}.json"
        
        self._get_load_spec_and_verify()
        self.receiver = WeightReceiver(controller_ipc_name, rank, self.load_spec.src_spec())
        
    def _get_hf_iter(self) -> Iterator[tuple[str, torch.Tensor]]:
        return self.loader._get_all_weights(self.model_config, self.model)

    def _get_load_spec_and_verify(self) -> None:
        """Load or infer ``self.load_spec`` and verify it against HF tensors and SGLang weights.

        Steps:

        1. If ``self.load_spec_path`` exists, parse it as a :class:`~wbridge.utils.data.LoadSpec` and
           :func:`~wbridge.utils.specgen.verify_load_spec`; return on success.
        2. On any failure (missing file, bad JSON, verification mismatch), log and infer a new spec:
           stream HF tensors, run a ``lw`` closure that mirrors HF→SGLang loading, then call
           :func:`~wbridge.utils.specgen.infer_load_spec` and verify again. Persist the result to
           ``self.load_spec_path``.
        """
        wksd = self.model.state_dict()
        try:
            with open(self.load_spec_path, encoding="utf-8") as f:
                self.load_spec = LoadSpec(json.load(f))
            verify_load_spec(self._get_hf_iter(), wksd, self.load_spec)
            return
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            
        self.load_spec = infer_load_spec(self._get_hf_iter(), wksd, self.model.load_weights)
        verify_load_spec(self._get_hf_iter(), wksd, self.load_spec)
        
        # Save the LoadSpec to cache
        os.makedirs(self.load_spec_path.parent, exist_ok=True)
        with open(self.load_spec_path, "w", encoding="utf-8") as f:
            json.dump(self.load_spec.entries, f, indent=2, sort_keys=True)
        logger.info("wbridge: wrote LoadSpec to %s", self.load_spec_path)
        return
    
    def is_weights_ready(self) -> bool:
        return self.receiver.is_weights_ready()
    