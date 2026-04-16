"""Framework frontends (Megatron-Bridge, SGLang integration)."""

from wbridge.frontend.megatron_adapter import WBMegatronAdapter
from wbridge.frontend.sglang_adapter import WBSGLangAdapter

__all__ = ["WBMegatronAdapter", "WBSGLangAdapter"]
