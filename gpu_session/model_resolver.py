"""Model resolution for HuggingFace models."""


class ModelResolver:
    """Resolve HuggingFace model specifications to SGLang-compatible IDs."""

    def parse_model_spec(self, model_spec: str) -> tuple[str, str, str]:
        """
        Parse a model specification into components.

        Args:
            model_spec: Model spec in format "org/repo" (e.g. "Qwen/Qwen3.6-27B-FP8")

        Returns:
            Tuple of (org, repo_name, "")
        """
        parts = model_spec.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1], ""
        return model_spec, "", ""

    def get_hf_repo_id(self, model_spec: str) -> str:
        """
        Get the HuggingFace repo ID from a model spec.

        Args:
            model_spec: Full model spec (e.g. "Qwen/Qwen3.6-27B-FP8")

        Returns:
            HuggingFace repo ID
        """
        return model_spec

    def get_hf_download_args(
        self,
        model_spec: str,
    ) -> tuple[str, str]:
        """
        Get model path for SGLang (returns repo ID directly, no GGUF download).

        Args:
            model_spec: Full model spec (e.g. "Qwen/Qwen3.6-27B-FP8")

        Returns:
            Tuple of (repo_id, "") — second element empty, kept for backward compat
        """
        return model_spec, ""
