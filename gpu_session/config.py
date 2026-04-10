"""Configuration management for gpu-session."""

import os
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import tomli_w


class Config:
    """Load and manage configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".gpu-session" / "config.toml"
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, "rb") as f:
                self._config = tomllib.load(f)

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "wb") as f:
            tomli_w.dump(self._config, f)

    @property
    def defaults(self) -> dict:
        """Get defaults section."""
        return self._config.get("defaults", {})

    @property
    def runpod_config(self) -> dict:
        """Get runpod section."""
        return self._config.get("runpod", {})

    @property
    def default_engine(self) -> str:
        """Get default inference engine."""
        return self.defaults.get("engine", "llama-cpp")

    @property
    def default_context_length(self) -> int:
        """Get default context length."""
        return self.defaults.get("context_length", 131072)

    @property
    def default_idle_timeout(self) -> int:
        """Get default idle timeout in minutes."""
        return self.defaults.get("idle_timeout", 0)

    @property
    def preferred_gpus(self) -> list[str]:
        """Get preferred GPU list."""
        return self.defaults.get("preferred_gpus", [])

    @property
    def cloud_type(self) -> str:
        """Get cloud type (SECURE or COMMUNITY)."""
        return self.runpod_config.get("cloud", "SECURE")


config = Config()
