"""Tests for model resolver."""

import pytest
from gpu_session.model_resolver import ModelResolver


class TestModelResolver:
    """Test model resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = ModelResolver()

    def test_parse_model_spec_with_quant(self):
        """Test parsing model spec with quantization."""
        repo, name, quant = self.resolver.parse_model_spec(
            "unsloth/Qwen3.5-27B-GGUF:Q8_0"
        )
        assert repo == "unsloth"
        assert name == "Qwen3.5-27B-GGUF"
        assert quant == "Q8_0"

    def test_parse_model_spec_without_quant(self):
        """Test parsing model spec without quantization."""
        repo, name, quant = self.resolver.parse_model_spec(
            "unsloth/Qwen3.5-27B-GGUF"
        )
        assert repo == "unsloth"
        assert name == "Qwen3.5-27B-GGUF"
        assert quant == ""

    def test_resolve_gguf_filename(self):
        """Test GGUF filename resolution."""
        filename = self.resolver.resolve_gguf_filename(
            "unsloth", "Qwen3.5-27B-GGUF", "Q8_0"
        )
        assert filename == "Qwen3.5-27B-Q8_0.gguf"

    def test_get_hf_download_args(self):
        """Test HuggingFace download args."""
        repo_id, filename = self.resolver.get_hf_download_args(
            "unsloth/Qwen3.5-27B-GGUF:Q8_0"
        )
        assert repo_id == "unsloth/Qwen3.5-27B-GGUF"
        assert filename == "Qwen3.5-27B-Q8_0.gguf"
