"""Framework frontends (Megatron-Bridge, SGLang integration)."""

"""Framework frontends.

Only framework-agnostic classes are re-exported here. The framework-specific adapters
(:class:`~wbridge.frontend.megatron_adapter.WBMegatronAdapter`,
:class:`~wbridge.frontend.sglang_adapter.WBSGLangAdapter`) must be imported from their respective
modules so importing this package does not transitively pull in heavy framework dependencies
(``megatron.bridge``, ``sglang``, ``vllm``, ...) on machines that don't have them installed.
"""

from wbridge.frontend.adapters import (
    AdapterContext,
    BaseAdapter,
    ReceiverAdapter,
    SenderAdapter,
)

__all__ = [
    "AdapterContext",
    "BaseAdapter",
    "ReceiverAdapter",
    "SenderAdapter",
]
