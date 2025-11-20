"""
FAISS-based semantic similarity wrapper dedicated to the new pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from .config import SemanticSearchConfig
from .inference import FaissMatcher

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchState:
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SemanticSearchEngine:
    """
    Global FAISS semantic search that always works in tandem with the tag classifier.
    """

    def __init__(self, config: Optional[SemanticSearchConfig] = None):
        self.config = config or SemanticSearchConfig()
        self._matcher: Optional[FaissMatcher] = None
        self._state: Optional[SemanticSearchState] = None

    def initialize(self):
        logger.info("Initializing semantic search engine (models_dir=%s)", self.config.models_dir)
        self._matcher = FaissMatcher(self.config)
        self._matcher.initialize()
        logger.info("✓ Semantic search ready")

    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[Any] = None
    ) -> SemanticSearchState:
        if not self._matcher:
            raise RuntimeError("SemanticSearchEngine not initialized")

        results, metadata = self._matcher.search_global(
            question,
            top_k or self.config.top_k,
            precomputed_embedding=precomputed_embedding
        )
        metadata = {**metadata, "strategy_used": "global"}
        state = SemanticSearchState(results=results, metadata=metadata)
        self._state = state
        return state
