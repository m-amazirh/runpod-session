"""Tests for model resolver."""

import pytest
from gpu_session.model_resolver import ModelResolver


class TestModelResolver:
    """Test model resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = ModelResolver()

    def test_parse_model_spec(self):
        """Test parsing model spec returns org and repo name."""
        org, name, _ = self.resolver.parse_model_spec(
            "Qwen/Qwen3.6-27B-FP8"
        )
        assert org == "Qwen"
        assert name == "Qwen3.6-27B-FP8"

    def test_parse_model_spec_no_slash(self):
        """Test parsing model spec without org."""
        org, name, _ = self.resolver.parse_model_spec(
            "Qwen3.6-27B-FP8"
        )
        assert org == "Qwen3.6-27B-FP8"
        assert name == ""

    def test_get_hf_repo_id(self):
        """Test get_hf_repo_id returns full spec as repo ID."""
        repo_id = self.resolver.get_hf_repo_id("Qwen/Qwen3.6-27B-FP8")
        assert repo_id == "Qwen/Qwen3.6-27B-FP8"

    def test_get_hf_download_args(self):
        """Test HuggingFace download args return repo ID directly."""
        repo_id, _ = self.resolver.get_hf_download_args(
            "Qwen/Qwen3.6-27B-FP8"
        )
        assert repo_id == "Qwen/Qwen3.6-27B-FP8"
