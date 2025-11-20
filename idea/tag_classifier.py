"""
Wrapper around the 203-tag UnifiedTagClassifier for the new pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from .config import TagClassifierConfig
from .inference import TagClassifierMatcher

logger = logging.getLogger(__name__)


@dataclass
class TagClassifierState:
    """Keeps the latest predictions and telemetry."""

    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class TagClassifier:
    """
    Thin wrapper on top of ``ClassifierMatcher``.

    Rationale: we want the rest of the ``idea`` package to treat the tag classifier
    as a black box that can be swapped out later (for instance if we train a new
    encoder or tweak the architecture described in ``training/train_tag_classifier.py``).
    """

    def __init__(self, config: Optional[TagClassifierConfig] = None):
        self.config = config or TagClassifierConfig()
        self._matcher: Optional[TagClassifierMatcher] = None
        self._state: Optional[TagClassifierState] = None

    def initialize(self):
        """Load weights, SentenceTransformer, and n-gram features."""
        logger.info("Initializing 203-tag classifier (models_dir=%s)", self.config.models_dir)
        self._matcher = TagClassifierMatcher(self.config)
        self._matcher.initialize()
        logger.info("✓ Tag classifier ready")

    def predict(
        self,
        question: str,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[Any] = None
    ) -> TagClassifierState:
        if not self._matcher:
            raise RuntimeError("TagClassifier not initialized")

        response = self._matcher.search(
            query=question,
            top_k=top_k or self.config.top_k,
            precomputed_embedding=precomputed_embedding
        )

        predictions = response.get("results", [])
        metadata = response.get("metadata", {})
        self._state = TagClassifierState(predictions=predictions, metadata=metadata)
        return self._state
