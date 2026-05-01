"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize configuration from file."""
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation (e.g., 'video.fps')."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation, preserving explicit null values."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                if k not in value:
                    return default
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.config[key]

    def __repr__(self) -> str:
        return f"Config({self.config_path})"
