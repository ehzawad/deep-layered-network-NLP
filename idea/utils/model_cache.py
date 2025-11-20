"""
Model Cache Utilities (idea version)

Copied from components.utils.model_cache so the experimental package is self-contained.
"""

import logging
import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_SHARED_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_CACHED_MODEL_NAME: Optional[str] = None
_CACHED_DEVICE: Optional[str] = None


def _mps_available() -> bool:
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _resolve_device_preference() -> Optional[str]:
    env_value = os.environ.get('STS_EMBEDDING_DEVICE')
    if not env_value:
        return None

    preferred = env_value.strip().lower()
    if preferred == 'auto':
        return None

    valid_options = {'cpu', 'cuda', 'mps'}
    if preferred not in valid_options:
        logger.warning(
            "STS_EMBEDDING_DEVICE=%s is invalid. Valid options: cpu, cuda, mps, auto. Falling back to auto.",
            preferred
        )
        return None

    if preferred == 'cuda' and not torch.cuda.is_available():
        logger.warning("STS_EMBEDDING_DEVICE=cuda but CUDA is unavailable. Falling back to auto selection.")
        return None

    if preferred == 'mps' and not _mps_available():
        logger.warning("STS_EMBEDDING_DEVICE=mps but MPS backend is unavailable. Falling back to auto selection.")
        return None

    return preferred


def _select_device() -> str:
    preferred = _resolve_device_preference()
    if preferred:
        return preferred

    if torch.cuda.is_available():
        return 'cuda'
    if _mps_available():
        return 'mps'
    return 'cpu'


def _load_model_on_device(model_name: str, device: str) -> tuple[SentenceTransformer, str]:
    try:
        return SentenceTransformer(model_name, device=device), device
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not (device == 'cuda' and 'out of memory' in str(exc).lower()):
            raise

        logger.warning(
            "CUDA OOM while loading %s on GPU. Falling back to CPU. Set STS_EMBEDDING_DEVICE=cpu to avoid GPU usage.",
            model_name
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return SentenceTransformer(model_name, device='cpu'), 'cpu'


def get_shared_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    force_reload: bool = False
) -> SentenceTransformer:
    global _SHARED_EMBEDDING_MODEL, _CACHED_MODEL_NAME, _CACHED_DEVICE

    needs_reload = (
        force_reload or
        _SHARED_EMBEDDING_MODEL is None or
        _CACHED_MODEL_NAME != model_name
    )

    if needs_reload:
        logger.info("Loading shared embedding model: %s", model_name)

        if _SHARED_EMBEDDING_MODEL is not None:
            logger.info("  Replacing existing model: %s", _CACHED_MODEL_NAME)

        requested_device = _select_device()
        logger.info("  Preferred device: %s", requested_device)

        model, actual_device = _load_model_on_device(model_name, requested_device)

        _SHARED_EMBEDDING_MODEL = model
        _CACHED_MODEL_NAME = model_name
        _CACHED_DEVICE = actual_device

        logger.info("  Model loaded successfully on %s", actual_device.upper())
        logger.info("  Model max sequence length (intrinsic): %s", _SHARED_EMBEDDING_MODEL.max_seq_length)
        logger.info("  Embedding dimension: %s", _SHARED_EMBEDDING_MODEL.get_sentence_embedding_dimension())
    else:
        logger.debug("Reusing cached embedding model: %s on %s", model_name, _CACHED_DEVICE)

    return _SHARED_EMBEDDING_MODEL


def clear_model_cache():
    global _SHARED_EMBEDDING_MODEL, _CACHED_MODEL_NAME, _CACHED_DEVICE

    if _SHARED_EMBEDDING_MODEL is not None:
        logger.info("Clearing cached embedding model: %s", _CACHED_MODEL_NAME)
        _SHARED_EMBEDDING_MODEL = None
        _CACHED_MODEL_NAME = None
        _CACHED_DEVICE = None
    else:
        logger.debug("No cached model to clear")


def is_model_cached() -> bool:
    return _SHARED_EMBEDDING_MODEL is not None


def get_cached_model_name() -> Optional[str]:
    return _CACHED_MODEL_NAME
