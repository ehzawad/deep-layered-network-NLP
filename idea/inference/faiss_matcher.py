"""
FAISS semantic search matcher copied into idea/ so the prototype is self-contained.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config import SemanticSearchConfig, E5PrefixConfig
from ..utils.model_cache import get_shared_embedding_model

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
_FAISS = None

if TYPE_CHECKING:  # pragma: no cover
    import faiss  # type: ignore


class FaissMatcher:
    """
    Semantic similarity search against FAISS indices.
    """

    def __init__(self, config: SemanticSearchConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)

        self.embedding_model = None
        self.indices = {}
        self.question_mapping = None
        self.tag_to_answer = {}

    def initialize(self):
        logger.info("Initializing FAISS matcher (idea)...")
        logger.info("  Loading embedding model: %s", self.config.embedding_model)
        logger.info("  E5 prefixes enabled: %s", E5PrefixConfig.USE_PREFIXES)
        if E5PrefixConfig.USE_PREFIXES:
            logger.info("  STS query prefix: '%s'", E5PrefixConfig.STS_QUERY_PREFIX)
        
        self.embedding_model = get_shared_embedding_model(self.config.embedding_model)

        self._load_faiss_indices()
        self._load_question_mapping()
        self._load_answers()
        logger.info("  ✓ FAISS matcher ready")

    def _lazy_import_faiss(self):
        global _FAISS
        if _FAISS is None:
            import faiss as _faiss_mod  # type: ignore
            _FAISS = _faiss_mod
        return _FAISS

    def _load_faiss_indices(self):
        similarity_dir = self.models_dir
        if not similarity_dir.exists():
            raise FileNotFoundError(
                f"FAISS models directory not found: {similarity_dir}. "
                "Copy indices into idea/models/semantic or update IdeaConfig.semantic.models_dir."
            )

        global_index = similarity_dir / "faiss_index_global.index"
        if not global_index.exists():
            raise FileNotFoundError(f"Global FAISS index missing: {global_index}")

        faiss_mod = self._lazy_import_faiss()
        self.indices['global'] = faiss_mod.read_index(str(global_index))
        logger.info("  Loaded global index with %d vectors", self.indices['global'].ntotal)

    def _load_question_mapping(self):
        mapping_file = self.models_dir / "question_mapping.csv"
        if not mapping_file.exists():
            raise FileNotFoundError(f"question_mapping.csv not found at {mapping_file}")

        self.question_mapping = pd.read_csv(mapping_file)
        logger.info("  Loaded question mapping (%d rows)", len(self.question_mapping))

    def _load_answers(self):
        # Prefer co-located datasets with the model artifacts, then fall back to
        # the repo-level datasets directory to avoid drift when users point
        # IdeaConfig.semantic.models_dir to a different location.
        candidate_paths = [
            self.models_dir.parent / "datasets" / "tag_to_answer.json",
            self.models_dir / "tag_to_answer.json",
            DATASETS_DIR / "tag_to_answer.json",
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.tag_to_answer = json.load(f)
                logger.info("  Loaded %d answers from %s", len(self.tag_to_answer), path)
                return

        raise FileNotFoundError(
            f"tag_to_answer.json not found in any of: {[str(p) for p in candidate_paths]}"
        )

    def _encode_query(self, query: str, precomputed_embedding: Optional[np.ndarray] = None) -> np.ndarray:
        # If pre-computed embedding provided, use it (skip encoding)
        if precomputed_embedding is not None:
            return precomputed_embedding.astype('float32').reshape(1, -1)

        # Otherwise, encode the query
        # Apply E5-instruct prefix
        query_prefixed = E5PrefixConfig.format_sts_query(query)

        embedding = self.embedding_model.encode(
            [query_prefixed],
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding[0].astype('float32').reshape(1, -1)

    def _search_index(
        self,
        query: str,
        faiss_index,
        result_indices: Optional[List[int]],
        top_k: int,
        source_index_name: str,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self._encode_query(query, precomputed_embedding=precomputed_embedding)
        similarities, indices = faiss_index.search(query_embedding, top_k)
        similarities = similarities[0]
        indices = indices[0]

        results = []
        for similarity, idx in zip(similarities, indices):
            original_idx = result_indices[idx] if result_indices else idx
            if original_idx >= len(self.question_mapping):
                continue

            row = self.question_mapping.iloc[original_idx]
            results.append({
                'question': row['question'],
                'tag': row['tag'],
                'similarity': float(similarity),
                'score': float(similarity),
                'answer': self.tag_to_answer.get(row['tag'], 'Answer not found'),
                'source_index': source_index_name
            })
        return results

    def search_global(
        self,
        query: str,
        top_k: int,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start = time.time()
        results = self._search_index(
            query,
            self.indices['global'],
            None,
            top_k,
            source_index_name='global',
            precomputed_embedding=precomputed_embedding
        )
        total_time = (time.time() - start) * 1000
        metadata = {
            'strategy_used': 'global',
            'indices_queried': ['global'],
            'num_vectors_searched': self.indices['global'].ntotal,
            'search_time_ms': round(total_time, 2),
            'num_results': len(results),
            'used_precomputed_embedding': precomputed_embedding is not None
        }
        return results, metadata
