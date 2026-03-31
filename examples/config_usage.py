"""
Example: Using the config loader directly in scripts
=====================================================

This example demonstrates how to access configuration values
programmatically from config.yaml.
"""
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.config_loader import get_config, get_config_value

# Load configuration once
config = get_config()

# Access simple values
random_seed = get_config_value(config, "random_seed", default=42)
print(f"Random seed: {random_seed}")

# Access nested values using dot notation
top_k = get_config_value(config, "retrieval.top_k", default=10)
print(f"Retrieval top-k: {top_k}")

# Access paths
corpus_path = get_config_value(config, "data.corpus_ar")
print(f"Corpus path: {corpus_path}")

# Access training parameters
n_estimators = get_config_value(config, "classifier_training.n_estimators", default=200)
print(f"Classifier n_estimators: {n_estimators}")

# Provide fallback for missing values
custom_value = get_config_value(config, "nonexistent.key", default="fallback_value")
print(f"Custom value with fallback: {custom_value}")
