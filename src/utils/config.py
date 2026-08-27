"""Configuration loader for Sentinel Medical AI."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "pipeline_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Singleton config
_config = None


def get_config(config_path: Optional[str] = None) -> dict:
    """Get cached configuration."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config
