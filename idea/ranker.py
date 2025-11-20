"""
Signal fusion and reranking logic for the experimental dual-leg pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


def _score_from_sts(candidate: Dict[str, Any]) -> float:
    if not candidate:
        return 0.0
    if candidate.get("score") is not None:
        return float(candidate["score"])
    if candidate.get("similarity") is not None:
        return float(candidate["similarity"])
    return 0.0


@dataclass
class RankerTelemetry:
    """Structured summary of what the ranker observed."""

    weights: Dict[str, float]
    min_score: float
    num_fused_tags: int
    dropped_below_threshold: int
    dropped_for_signal_floor: int
    abstained: bool
    input_counts: Dict[str, int]


@dataclass
class RankerConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.6, "classifier": 0.4})
    min_score: float = 0.05
    min_sts_similarity: float = 0.2
    min_classifier_confidence: float = 0.2
    abstain_answer: str = "I can't answer that with enough confidence right now."
    abstain_on_low_signals: bool = True


class DualSignalRanker:
    """
    Blend STS similarity scores with classifier probabilities and surface the best tag.
    """

    def __init__(self, config: Optional[RankerConfig] = None):
        self.config = config or RankerConfig()
        self.weights = self._normalize(self.config.weights)

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, w) for w in weights.values())
        if total == 0:
            return {"sts": 0.5, "classifier": 0.5}
        return {key: max(0.0, value) / total for key, value in weights.items()}

    @staticmethod
    def _normalize_per_query_minmax(values: List[float]) -> List[float]:
        """
        Normalize scores using per-query min-max normalization.

        This ensures all scores are mapped to [0, 1] based on the min and max
        within the current query's results. This makes different signals (STS,
        classifier) comparable and allows fair weighted fusion.

        Args:
            values: Raw scores to normalize

        Returns:
            Normalized scores in [0, 1] where min→0, max→1

        Example:
            values = [0.75, 0.85, 0.90, 0.80]
            min=0.75, max=0.90, range=0.15
            normalized = [0.0, 0.67, 1.0, 0.33]
        """
        if not values:
            return []

        if len(values) == 1:
            return [1.0]  # Single value gets max score

        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero if all values are identical
        if max_val == min_val:
            return [1.0] * len(values)

        range_val = max_val - min_val
        normalized = [(v - min_val) / range_val for v in values]

        return normalized

    def _fuse(self, sts_results: List[Dict[str, Any]], clf_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}

        # Normalize STS scores using per-query min-max
        sts_raw_scores = [_score_from_sts(item) for item in sts_results]
        normalized_sts = self._normalize_per_query_minmax(sts_raw_scores)

        # Normalize classifier confidence scores using per-query min-max
        clf_raw_scores = [float(item.get("confidence", 0.0)) for item in clf_results]
        normalized_clf = self._normalize_per_query_minmax(clf_raw_scores)

        # Process STS results
        for idx, candidate in enumerate(sts_results):
            tag = candidate.get("tag")
            if not tag:
                continue
            node = fused.setdefault(tag, {"tag": tag, "answer": candidate.get("answer")})
            best = node.get("sts")
            normalized_score = normalized_sts[idx] if idx < len(normalized_sts) else 0.0
            if best is None or normalized_score > best.get("normalized_score", -1):
                node["sts"] = {
                    "raw_score": sts_raw_scores[idx] if idx < len(sts_raw_scores) else 0.0,
                    "normalized_score": normalized_score,
                    "rank": idx,
                    "source_index": candidate.get("source_index"),
                    "question": candidate.get("question"),
                    "similarity": candidate.get("similarity")
                }
            node.setdefault("answer", candidate.get("answer"))

        # Process classifier results
        for idx, candidate in enumerate(clf_results):
            tag = candidate.get("tag")
            if not tag:
                continue
            node = fused.setdefault(tag, {"tag": tag, "answer": candidate.get("answer")})
            best = node.get("classifier")
            raw_confidence = clf_raw_scores[idx] if idx < len(clf_raw_scores) else 0.0
            normalized_confidence = normalized_clf[idx] if idx < len(normalized_clf) else 0.0
            if best is None or normalized_confidence > best.get("normalized_confidence", -1):
                node["classifier"] = {
                    "raw_confidence": raw_confidence,
                    "normalized_confidence": normalized_confidence,
                    "rank": idx,
                    "source": candidate.get("source", "classifier")
                }
            node.setdefault("answer", candidate.get("answer"))

        return fused

    def _score_entry(self, entry: Dict[str, Any]) -> Tuple[float, List[Dict[str, float]], str]:
        contributions: List[Dict[str, float]] = []
        score_sum = 0.0
        total_weight = 0.0
        present: List[str] = []

        if entry.get("sts"):
            weight = self.weights.get("sts", 0.0)
            value = entry["sts"].get("normalized_score", 0.0)
            contributions.append({"signal": "sts", "weight": weight, "value": value})
            score_sum += weight * value
            total_weight += weight
            present.append("sts")

        if entry.get("classifier"):
            weight = self.weights.get("classifier", 0.0)
            value = entry["classifier"].get("normalized_confidence", 0.0)
            contributions.append({"signal": "classifier", "weight": weight, "value": value})
            score_sum += weight * value
            total_weight += weight
            present.append("classifier")

        if total_weight == 0:
            return 0.0, contributions, "none"

        final_score = score_sum / total_weight
        source = "+".join(present) if present else "none"
        return final_score, contributions, source

    def rank(
        self,
        sts_results: Optional[List[Dict[str, Any]]],
        classifier_results: Optional[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], RankerTelemetry, List[Dict[str, Any]]]:
        sts_results = sts_results or []
        classifier_results = classifier_results or []

        fused_entries = self._fuse(sts_results, classifier_results)
        ranked: List[Dict[str, Any]] = []
        dropped_candidates: List[Dict[str, Any]] = []
        dropped = 0
        signal_floor_drops = 0

        for entry in fused_entries.values():
            final_score, contributions, source = self._score_entry(entry)
            raw_sts = (entry.get("sts") or {}).get("raw_score")
            raw_clf = (entry.get("classifier") or {}).get("raw_confidence")
            below_sts_floor = (
                self.config.abstain_on_low_signals
                and self.config.min_sts_similarity is not None
                and (raw_sts is None or raw_sts < self.config.min_sts_similarity)
            )
            below_clf_floor = (
                self.config.abstain_on_low_signals
                and self.config.min_classifier_confidence is not None
                and (raw_clf is None or raw_clf < self.config.min_classifier_confidence)
            )
            node = {
                "tag": entry["tag"],
                "answer": entry.get("answer"),
                "final_score": final_score,
                "contributions": contributions,
                "source": source,
                "signals": {
                    "sts": entry.get("sts"),
                    "classifier": entry.get("classifier")
                }
            }
            if below_sts_floor and below_clf_floor:
                signal_floor_drops += 1
                node["drop_reason"] = "signal_floor"
                dropped_candidates.append(node)
                continue

            if final_score < self.config.min_score:
                dropped += 1
                node["drop_reason"] = "below_min_score"
                dropped_candidates.append(node)
                continue
            ranked.append(node)

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        dropped_candidates.sort(key=lambda item: item["final_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        telemetry = RankerTelemetry(
            weights=self.weights,
            min_score=self.config.min_score,
            num_fused_tags=len(fused_entries),
            dropped_below_threshold=dropped,
            dropped_for_signal_floor=signal_floor_drops,
            abstained=len(ranked) == 0,
            input_counts={"sts": len(sts_results), "classifier": len(classifier_results)}
        )

        return ranked, telemetry, dropped_candidates
