"""
config_loader.py — Configuration file loader for the Doha Dictionary RAG system.

Loads settings from ``config.yaml`` at the repository root and provides
convenient access to nested configuration values.

Usage::

    from src.config_loader import load_config, get_config_value

    # Load once at module level
    config = load_config()

    # Access nested values
    top_k = get_config_value(config, "retrieval.top_k", default=10)
    corpus_path = get_config_value(config, "data.corpus_ar")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, searches for config.yaml
                     in the repository root (parent directory of src/).

    Returns:
        Dictionary containing all configuration values.
        Returns empty dict if file not found (graceful degradation).
    """
    if config_path is None:
        # Auto-detect: assume this file is in src/, so repo root is parent
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        # Graceful degradation: return empty config if file missing
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """Retrieve a nested configuration value using dot notation.

    Args:
        config:   Configuration dictionary (output of load_config).
        key_path: Dot-separated path to the value (e.g., "retrieval.top_k").
        default:  Value to return if key_path is not found.

    Returns:
        The configuration value, or *default* if not found.

    Examples:
        >>> config = {"retrieval": {"top_k": 10, "method": "hybrid"}}
        >>> get_config_value(config, "retrieval.top_k")
        10
        >>> get_config_value(config, "retrieval.missing", default=5)
        5
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


# ────────────────────────────────────────────────────────────────────────── #
# Singleton instance — load once at import time                               #
# ────────────────────────────────────────────────────────────────────────── #

_CONFIG: Optional[dict[str, Any]] = None


def get_config() -> dict[str, Any]:
    """Return the singleton configuration dictionary.

    The config is loaded once on first call and cached for subsequent calls.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG
