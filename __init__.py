"""ComfyUI-Vates-Sampler：V-Sampler 节点注册."""

from __future__ import annotations

from .v_sampler_node import VatesAdvancedSampler

NODE_CLASS_MAPPINGS = {"VatesAdvancedSampler": VatesAdvancedSampler}

NODE_DISPLAY_NAME_MAPPINGS = {"VatesAdvancedSampler": "V-Sampler"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
