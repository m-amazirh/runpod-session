"""Model resolution for GGUF models on HuggingFace."""

import re
from typing import Optional


class ModelResolver:
    """Resolve HuggingFace model specifications to GGUF filenames."""

    def __init__(self):
        # Common quantization suffix patterns
        self.quant_patterns = {
            "Q8_0": "Q8_0",
            "Q5_K_M": "Q5_K_M",
            "Q4_K_M": "Q4_K_M",
            "Q4_0": "Q4_0",
            "Q2_K": "Q2_K",
            "IQ4_XS": "IQ4_XS",
            "IQ3_XXS": "IQ3_XXS",
        }

    def parse_model_spec(self, model_spec: str) -> tuple[str, str, str]:
        """
        Parse a model specification into components.

        Args:
            model_spec: Model spec in format "repo/name:quantization"
                       or just "repo/name" (will find latest GGUF)

        Returns:
            Tuple of (repo, model_name, quantization)
        """
        # Split by colon for quantization
        if ":" in model_spec:
            base, quant = model_spec.rsplit(":", 1)
        else:
            base = model_spec
            quant = ""

        # Split repo/name
        parts = base.split("/", 1)
        if len(parts) == 2:
            repo, model_name = parts
        else:
            repo, model_name = base, ""

        return repo, model_name, quant

    def resolve_gguf_filename(
        self,
        repo: str,
        model_name: str,
        quantization: str,
    ) -> str:
        """
        Resolve the GGUF filename for a model.

        Args:
            repo: HuggingFace repo (e.g., "unsloth")
            model_name: Model name (e.g., "Qwen3.5-27B-GGUF")
            quantization: Quantization suffix (e.g., "Q8_0")

        Returns:
            GGUF filename (e.g., "Qwen3.5-27B-Q8_0.gguf")
        """
        # Remove -GGUF suffix from model name for filename
        base_name = model_name.replace("-GGUF", "")

        if quantization:
            return f"{base_name}-{quantization}.gguf"

        # If no quantization specified, return pattern for latest
        return f"{base_name}-*.gguf"

    def get_hf_download_args(
        self,
        model_spec: str,
    ) -> tuple[str, str]:
        """
        Get HuggingFace CLI download arguments.

        Args:
            model_spec: Full model spec (e.g., "unsloth/Qwen3.5-27B-GGUF:Q8_0")

        Returns:
            Tuple of (repo_id, filename_pattern)
        """
        repo, model_name, quant = self.parse_model_spec(model_spec)
        repo_id = f"{repo}/{model_name}"
        filename = self.resolve_gguf_filename(repo, model_name, quant)

        return repo_id, filename
