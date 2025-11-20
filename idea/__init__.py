"""
Public entry points for the experimental dual-signal pipeline.

Imports are performed lazily so that heavy dependencies (faiss, torch, etc.)
are only pulled in when the corresponding attribute is accessed. This keeps
utility modules usable in isolation (e.g., idea.utils.model_cache) without
triggering faiss → SentenceTransformer segfaults on macOS.
"""

from typing import TYPE_CHECKING

__all__ = ["IdeaConfig", "IdeaPipeline"]

if TYPE_CHECKING:  # pragma: no cover
    from .config import IdeaConfig as _IdeaConfig
    from .pipeline import IdeaPipeline as _IdeaPipeline


def __getattr__(name: str):
    if name == "IdeaConfig":
        from .config import IdeaConfig as _IdeaConfig  # local import to avoid eager deps
        return _IdeaConfig
    if name == "IdeaPipeline":
        from .pipeline import IdeaPipeline as _IdeaPipeline
        return _IdeaPipeline
    raise AttributeError(f"module 'idea' has no attribute {name!r}")
