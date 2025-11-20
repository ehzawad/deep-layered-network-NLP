"""
Configuration objects for the experimental dual-signal pipeline that lives
inside the ``idea/`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ============================================================================
# GLOBAL MODEL CONFIGURATION
# ============================================================================

# Single source of truth for the embedding model used throughout the pipeline
# Change this value to use a different embedding model across all components
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"


# ============================================================================
# E5-INSTRUCT PREFIX CONFIGURATION
# ============================================================================

class E5PrefixConfig:
    """
    Centralized configuration for E5-instruct model prefixes.
    
    The intfloat/multilingual-e5-large-instruct model requires specific prefixes
    for optimal performance. This class centralizes all prefix logic.
    
    Usage:
        # For STS semantic search
        query_text = E5PrefixConfig.format_sts_query("How do I reset password?")
        passage_text = E5PrefixConfig.format_sts_passage("Reset password guide")
        
        # For tag classification
        query_text = E5PrefixConfig.format_classifier_query("How do I reset password?")
    """
    
    # Enable/disable prefixes globally
    USE_PREFIXES: bool = True
    
    # Simple prefix mode (Option A - recommended)
    STS_QUERY_PREFIX: str = "query: "
    STS_PASSAGE_PREFIX: str = "passage: "
    CLASSIFIER_QUERY_PREFIX: str = "query: "
    
    # Alternative: Task-specific instruction mode (Option B - more advanced)
    # Uncomment these and set USE_TASK_INSTRUCTIONS = True to use
    # USE_TASK_INSTRUCTIONS: bool = False
    # STS_TASK_INSTRUCTION: str = "Retrieve similar customer questions about National ID card services"
    # CLASSIFIER_TASK_INSTRUCTION: str = "Classify this National ID support question into the appropriate category"
    
    @classmethod
    def format_sts_query(cls, text: str) -> str:
        """Format user query for STS semantic search."""
        if not cls.USE_PREFIXES:
            return text
        return cls.STS_QUERY_PREFIX + text
    
    @classmethod
    def format_sts_passage(cls, text: str) -> str:
        """Format training corpus passage for STS semantic search."""
        if not cls.USE_PREFIXES:
            return text
        return cls.STS_PASSAGE_PREFIX + text
    
    @classmethod
    def format_sts_passages_batch(cls, texts: List[str]) -> List[str]:
        """Format batch of passages (more efficient than loop)."""
        if not cls.USE_PREFIXES:
            return texts
        return [cls.STS_PASSAGE_PREFIX + t for t in texts]
    
    @classmethod
    def format_classifier_query(cls, text: str) -> str:
        """Format query for tag classifier (both training and inference)."""
        if not cls.USE_PREFIXES:
            return text
        return cls.CLASSIFIER_QUERY_PREFIX + text
    
    @classmethod
    def format_classifier_queries_batch(cls, texts: List[str]) -> List[str]:
        """Format batch of queries for classifier (more efficient than loop)."""
        if not cls.USE_PREFIXES:
            return texts
        return [cls.CLASSIFIER_QUERY_PREFIX + t for t in texts]
    
    @classmethod
    def get_metadata(cls) -> Dict[str, any]:
        """Get prefix configuration as metadata for saving to artifacts."""
        return {
            'use_prefixes': cls.USE_PREFIXES,
            'sts_query_prefix': cls.STS_QUERY_PREFIX if cls.USE_PREFIXES else None,
            'sts_passage_prefix': cls.STS_PASSAGE_PREFIX if cls.USE_PREFIXES else None,
            'classifier_query_prefix': cls.CLASSIFIER_QUERY_PREFIX if cls.USE_PREFIXES else None,
        }
    
    @classmethod
    def get_cache_key(cls) -> str:
        """Get cache key component for fingerprinting."""
        import json
        return json.dumps({
            'use_prefixes': cls.USE_PREFIXES,
            'sts_query_prefix': cls.STS_QUERY_PREFIX,
            'sts_passage_prefix': cls.STS_PASSAGE_PREFIX,
            'classifier_query_prefix': cls.CLASSIFIER_QUERY_PREFIX,
        }, sort_keys=True)


def _normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _as_path(value: Optional[str | Path], default: Path) -> Path:
    raw = default if value is None else Path(value)
    return _normalize_path(raw)


@dataclass
class SemanticSearchConfig:
    """
    Settings for the FAISS/STS semantic search leg.

    Note: E5-instruct prefix handling is centralized in E5PrefixConfig class above.
    """

    models_dir: Path = Path("idea/models/semantic")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 5
    strategy: str = "global"  # Always run global for the tandem workflow
    normalize_embeddings: bool = True

    def __post_init__(self):
        self.models_dir = _as_path(self.models_dir, Path("idea/models/semantic"))


@dataclass
class TagClassifierConfig:
    """
    Settings for the 203-tag classifier leg.

    Note: E5-instruct prefix handling is centralized in E5PrefixConfig class above.

    By default we reuse the UnifiedTagClassifier artifacts produced by the
    legacy STS classifier training recipe.
    """

    models_dir: Path = Path("idea/models/tag_classifier")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 25
    ngram_mode: str = "manual"  # manual / auto features for pattern branch
    use_token_attention: bool = False  # Use token-level attention (experimental)
    token_attention_variant: str = "lightweight"  # lightweight / multihead

    def __post_init__(self):
        self.models_dir = _as_path(self.models_dir, Path("idea/models/tag_classifier"))


@dataclass
class RankerConfig:
    """Weights + filtering for the final decider."""

    # weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.75, "classifier": 0.25})
    # weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.49, "classifier": 0.51})
    # weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0, "classifier": 1})
    weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.20, "classifier": 0.80})
    # weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.35, "classifier": 0.65})
    min_score: float = 0.05
    # min_score: float = 0.45
    # Hard floors (raw, pre-normalization) for both signals; if both are below their
    # respective floors the candidate is dropped, even if min-max normalization would
    # inflate it.
    min_sts_similarity: float = 0.2
    min_classifier_confidence: float = 0.2
    # When all candidates are dropped we surface this abstain answer instead of
    # returning a low-confidence tag.
    abstain_answer: str = "I can't answer that with enough confidence right now."
    abstain_on_low_signals: bool = True


@dataclass
class IdeaConfig:
    """
    Aggregate config consumed by the experimental pipeline.

    Attributes:
        semantic: SemanticSearchConfig
        classifier: TagClassifierConfig
        ranker: RankerConfig
        fusion_top_k: How many fused answers to keep before returning top-1
        log_dir: Optional location for structured logs/metrics
    """

    semantic: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    classifier: TagClassifierConfig = field(default_factory=TagClassifierConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    fusion_top_k: int = 10
    log_dir: Optional[Path] = None

    def __post_init__(self):
        if self.log_dir is not None:
            self.log_dir = _as_path(self.log_dir, Path("idea/logs"))

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "IdeaConfig":
        if not data:
            return cls()

        semantic = data.get("semantic")
        classifier = data.get("classifier")
        ranker = data.get("ranker")

        return cls(
            semantic=SemanticSearchConfig(**semantic) if semantic else SemanticSearchConfig(),
            classifier=TagClassifierConfig(**classifier) if classifier else TagClassifierConfig(),
            ranker=RankerConfig(**ranker) if ranker else RankerConfig(),
            fusion_top_k=data.get("fusion_top_k", cls.fusion_top_k),
            log_dir=data.get("log_dir")
        )
