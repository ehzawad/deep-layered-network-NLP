"""
IDEA Featurizer Package

Provides n-gram extraction, analysis, and cleanup utilities for the 203-tag
classifier. Outputs are stored under idea/datasets/features/.
"""

from pathlib import Path

FEATURIZER_DIR = Path(__file__).parent
FEATURES_DIR = Path(__file__).parent.parent / "datasets" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["FEATURIZER_DIR", "FEATURES_DIR"]
