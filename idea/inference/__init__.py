"""
Inference Module for pipelineNLP

This module contains all runtime inference components:
- Model architectures (UnifiedTagClassifier)
- Inference engines (FaissMatcher, TagClassifierMatcher)

Designed for easy import in deployment scenarios (FastAPI, microservices, etc.)

Usage:
    from idea.inference import FaissMatcher, TagClassifierMatcher, UnifiedTagClassifier
"""

from .unified_tag_classifier import UnifiedTagClassifier
from .faiss_matcher import FaissMatcher
from .tag_classifier_matcher import TagClassifierMatcher

__all__ = ['UnifiedTagClassifier', 'FaissMatcher', 'TagClassifierMatcher']
