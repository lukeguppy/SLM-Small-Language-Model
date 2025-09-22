"""
Core modules for the SLM project.

This package contains the core functionality for model training,
data processing, and utilities.
"""

# Avoid circular imports by not importing classes that depend on model_config
# Import them lazily when needed
from .utils import softmax, cross_entropy_loss, sgd_update, perplexity
from .embedding_utils import reduce_embeddings_2d
from .paths import ProjectPaths, ModelPaths, DataPaths
from .model_discovery import ModelDiscovery
from .file_ops import FileOperations

__all__ = [
    "softmax",
    "cross_entropy_loss",
    "sgd_update",
    "perplexity",
    "reduce_embeddings_2d",
    "ProjectPaths",
    "ModelPaths",
    "DataPaths",
    "ModelDiscovery",
    "FileOperations",
]
