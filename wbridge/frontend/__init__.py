"""Framework frontends (Megatron-Bridge, SGLang integration)."""

from wbridge.frontend.adapters import (
    AdapterContext,
    BaseAdapter,
    ReceiverAdapter,
    SenderAdapter,
)
from wbridge.frontend.megatron_adapter import WBMegatronAdapter
from wbridge.frontend.sglang_adapter import WBSGLangAdapter

__all__ = [
    "AdapterContext",
    "BaseAdapter",
    "ReceiverAdapter",
    "SenderAdapter",
    "WBMegatronAdapter",
    "WBSGLangAdapter",
]
