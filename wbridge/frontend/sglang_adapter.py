"""SGLang frontend for WeightBridge.

Builds ``hf_iter_factory`` from SGLang's ``ModelRunner`` loader, exposes ``model.state_dict()`` as
``wksd``, and uses ``model.load_weights`` for spec inference. LoadSpec lifecycle and
:class:`~wbridge.backend.receiver.WeightReceiver` plumbing are inherited from
:class:`~wbridge.frontend.adapters.ReceiverAdapter`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch
from sglang.srt.model_executor.model_runner import ModelRunner

from wbridge.frontend.adapters import AdapterContext, ReceiverAdapter

logger = logging.getLogger(__name__)


class WBSGLangAdapter(ReceiverAdapter):
    """Receive weights via :class:`~wbridge.frontend.adapters.ReceiverAdapter` into SGLang's
    ``model.state_dict()``.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        rank: int,
        controller_ipc_name: str,
    ) -> None:
        loader = model_runner.loader
        model_config = model_runner.model_config
        model = model_runner.model

        def hf_iter_factory() -> Iterator[tuple[str, torch.Tensor]]:
            return loader._get_all_weights(model_config, model)

        ctx = AdapterContext(
            hf_iter_factory=hf_iter_factory,
            wksd=model.state_dict(),
            load_weights=model.load_weights,
            load_spec_path=Path.home() / ".cache" / "sglang" / f"loadspec_rank{rank}.json",
            rank=rank,
        )
        super().__init__(ctx, controller_ipc_name)
