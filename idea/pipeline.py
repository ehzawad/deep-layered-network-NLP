"""
Experimental dual-leg pipeline that runs STS similarity + 203-tag classification
in tandem and relies on a ranker/decider to pick the best answer.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .config import IdeaConfig, E5PrefixConfig
from .semantic_similarity import SemanticSearchEngine
from .tag_classifier import TagClassifier
from .ranker import DualSignalRanker, RankerConfig as RankerRuntimeConfig
from .utils.model_cache import get_shared_embedding_model

logger = logging.getLogger(__name__)


class IdeaPipeline:
    """
    pipelineNLP runner (new dual-signal pipeline) that lives entirely under
    ``idea/`` so legacy components remain untouched.
    """

    def __init__(self, config: Optional[IdeaConfig | Dict[str, Any]] = None):
        if isinstance(config, IdeaConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = IdeaConfig.from_dict(config)
        else:
            self.config = IdeaConfig()

        self.semantic = SemanticSearchEngine(self.config.semantic)
        self.classifier = TagClassifier(self.config.classifier)

        # Initialize ranker with per-query min-max normalization (no percentiles needed)
        self.ranker = DualSignalRanker(
            RankerRuntimeConfig(
                weights=self.config.ranker.weights,
                min_score=self.config.ranker.min_score,
                min_sts_similarity=self.config.ranker.min_sts_similarity,
                min_classifier_confidence=self.config.ranker.min_classifier_confidence,
                abstain_answer=self.config.ranker.abstain_answer,
                abstain_on_low_signals=self.config.ranker.abstain_on_low_signals,
            )
        )

        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.info("pipelineNLP already initialized")
            return

        logger.info("Initializing pipelineNLP (dual STS + 203-tag classifier)...")
        self.semantic.initialize()
        self.classifier.initialize()
        self._initialized = True
        logger.info("✓ pipelineNLP ready")

    def run(
        self,
        question: str,
        fusion_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        fusion_top_k = fusion_top_k or self.config.fusion_top_k
        start_time = time.time()

        # OPTIMIZATION: Encode once only if both legs truly share model, prefix, and
        # normalization settings; otherwise let each matcher encode independently to
        # avoid silent prefix/model drift.
        shared_embedding = None
        shared_embedding_used = False
        if self._can_share_embedding():
            model = get_shared_embedding_model(self.config.semantic.embedding_model)
            question_prefixed = E5PrefixConfig.format_sts_query(question)
            shared_embedding = model.encode(
                [question_prefixed],
                normalize_embeddings=self.config.semantic.normalize_embeddings,
                show_progress_bar=False
            )[0]
            shared_embedding_used = True

        # Leg 1 – STS similarity (global) - pass pre-computed embedding
        sts_state = self.semantic.search(
            question,
            top_k=self.config.semantic.top_k,
            precomputed_embedding=shared_embedding
        )

        # Leg 2 – Tag classifier (203 tags) - pass pre-computed embedding
        clf_state = self.classifier.predict(
            question,
            top_k=self.config.classifier.top_k,
            precomputed_embedding=shared_embedding
        )

        # Ranker – fuse signals
        ranked, telemetry, dropped_candidates = self.ranker.rank(
            sts_state.results,
            clf_state.predictions,
            top_k=fusion_top_k
        )

        best = ranked[0] if ranked else None
        primary_answer = (
            (best or {}).get("answer")
            or (clf_state.predictions[0]["answer"] if clf_state.predictions else None)
            or (sts_state.results[0]["answer"] if sts_state.results else None)
            or "Answer not found"
        )
        if telemetry.abstained and self.config.ranker.abstain_answer:
            primary_answer = self.config.ranker.abstain_answer

        total_time = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": primary_answer,
            "primary_tag": (best or {}).get("tag"),
            "candidates": ranked,
            "dropped_candidates": dropped_candidates,
            "signals": {
                "sts": {
                    "results": sts_state.results,
                    "metadata": sts_state.metadata
                },
                "classifier": {
                    "results": clf_state.predictions,
                    "metadata": clf_state.metadata
                }
            },
            "telemetry": {
                "ranker": telemetry.__dict__,
                "latency_ms": round(total_time, 2),
                "fusion_top_k": fusion_top_k,
                "shared_embedding_used": shared_embedding_used
            }
        }

    def _can_share_embedding(self) -> bool:
        """Only reuse embeddings when model/prefix/normalization choices align."""
        same_model = self.config.semantic.embedding_model == self.config.classifier.embedding_model
        same_prefix = (
            not E5PrefixConfig.USE_PREFIXES or
            E5PrefixConfig.STS_QUERY_PREFIX == E5PrefixConfig.CLASSIFIER_QUERY_PREFIX
        )
        return (
            same_model and
            same_prefix and
            self.config.semantic.normalize_embeddings
        )
