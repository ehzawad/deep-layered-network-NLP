"""
Classifier matcher dedicated to the idea pipeline.

Forked from components.sts.inference.classifier_matcher to keep the experimental
codebase self-contained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.model_cache import get_shared_embedding_model

from ..config import TagClassifierConfig, E5PrefixConfig
from .unified_tag_classifier import UnifiedTagClassifier
from .token_attention_classifier import LightweightTokenClassifier, MultiHeadTokenClassifier
from ..utils.token_utils import extract_token_embeddings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
FEATURES_DIR = DATASETS_DIR / "features"


class TagClassifierMatcher:
    """Predicts tags using the UnifiedTagClassifier."""

    def __init__(self, config: TagClassifierConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.embedding_model_name = config.embedding_model
        self.use_token_attention = config.use_token_attention
        self.token_attention_variant = config.token_attention_variant

        self.device = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.model: Optional[UnifiedTagClassifier] = None

        self.tag_encoder = None
        self.pattern_mean = None
        self.pattern_std = None
        self.tag_patterns = None
        self.tags_sorted = None

        self.tag_to_answer: Dict[str, str] = {}

    def initialize(self):
        logger.info("Loading tag classifier matcher from %s", self.models_dir)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info("  Device: %s", self.device)

        logger.info("  Loading embedding model: %s", self.embedding_model_name)
        logger.info("  E5 prefixes enabled: %s", E5PrefixConfig.USE_PREFIXES)
        if E5PrefixConfig.USE_PREFIXES:
            logger.info("  Classifier query prefix: '%s'", E5PrefixConfig.CLASSIFIER_QUERY_PREFIX)
        
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)

        # Determine which model file to load
        if self.use_token_attention:
            model_file = self.models_dir / f"token_attention_{self.token_attention_variant}.pth"
        else:
            model_file = self.models_dir / "unified_tag_classifier.pth"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {model_file}. "
                "Run idea/training/train_tag_classifier.py first."
            )

        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        num_tags = checkpoint['num_tags']
        self.tag_encoder = checkpoint['tag_encoder_classes']

        logger.info("  Tags: %d", num_tags)

        # Token attention models don't use n-gram patterns
        if self.use_token_attention:
            logger.info("  Model type: Token Attention (%s)", self.token_attention_variant)

            # Extract architecture config from checkpoint
            token_dim = checkpoint.get('token_dim', 1024)
            dropout = checkpoint.get('dropout', 0.5)

            if self.token_attention_variant == "lightweight":
                hidden_dim = checkpoint.get('hidden_dim', 256)
                logger.info("    - Token dim: %d", token_dim)
                logger.info("    - Hidden dim: %d", hidden_dim)
                logger.info("    - Dropout: %.2f", dropout)

                self.model = LightweightTokenClassifier(
                    token_dim=token_dim,
                    num_tags=num_tags,
                    dropout=dropout,
                    hidden_dim=hidden_dim
                ).to(self.device)
            else:  # multihead
                num_heads = checkpoint.get('num_heads', 8)
                logger.info("    - Token dim: %d", token_dim)
                logger.info("    - Num heads: %d", num_heads)
                logger.info("    - Dropout: %.2f", dropout)

                self.model = MultiHeadTokenClassifier(
                    token_dim=token_dim,
                    num_tags=num_tags,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

        else:
            # Legacy unified classifier with n-gram patterns
            self.pattern_mean = np.array(checkpoint['pattern_mean'], dtype=np.float32).squeeze()
            self.pattern_std = np.array(checkpoint['pattern_std'], dtype=np.float32).squeeze()

            if self.pattern_mean.ndim != 1:
                self.pattern_mean = self.pattern_mean.reshape(-1)
            if self.pattern_std.ndim != 1:
                self.pattern_std = self.pattern_std.reshape(-1)

            zero_std = self.pattern_std == 0
            if np.any(zero_std):
                logger.warning(
                    "Pattern std contained %d zeros; replacing with 1.0",
                    int(np.sum(zero_std))
                )
                self.pattern_std[zero_std] = 1.0

            self._load_ngram_patterns(self.config.ngram_mode)

            pattern_dim = len(self.tags_sorted) * 5  # uni + bi + tri + four + five grams
            logger.info("  Pattern dim: %d (5 types × %d tags)", pattern_dim, len(self.tags_sorted))

            # Extract architecture config from checkpoint (for backward compatibility)
            # Old models won't have these keys, so we default to legacy architecture
            use_film = checkpoint.get('use_film', False)
            use_auxiliary = checkpoint.get('use_auxiliary', False)
            pattern_output_dim = checkpoint.get('pattern_output_dim', 64)  # Legacy: 64, FiLM: 128
            dropout = checkpoint.get('dropout', 0.5)

            logger.info("  Model architecture:")
            logger.info("    - FiLM conditioning: %s", use_film)
            logger.info("    - Auxiliary heads: %s", use_auxiliary)
            logger.info("    - Pattern output dim: %d", pattern_output_dim)
            logger.info("    - Dropout: %.2f", dropout)

            self.model = UnifiedTagClassifier(
                embedding_dim=self.embedding_model.get_sentence_embedding_dimension(),
                pattern_dim=pattern_dim,
                num_tags=num_tags,
                dropout=dropout,
                pattern_output_dim=pattern_output_dim,
                use_film=use_film,
                use_auxiliary=use_auxiliary
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

        self._load_answers()
        logger.info("  ✓ Tag classifier matcher ready")

    def _load_answers(self):
        path_options = [
            self.models_dir / "tag_to_answer.json",
            self.models_dir.parent / "datasets" / "tag_to_answer.json",
            DATASETS_DIR / "tag_to_answer.json",
        ]
        for path in path_options:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.tag_to_answer = json.load(f)
                logger.info("  Loaded %d answers from %s", len(self.tag_to_answer), path)
                return
        raise FileNotFoundError("tag_to_answer.json not found in idea datasets or model directory.")

    def _load_ngram_patterns(self, mode: str):
        repo_path = FEATURES_DIR / f"{mode}_ngrams.json"
        co_located_path = self.models_dir.parent / "datasets" / "features" / f"{mode}_ngrams.json"
        path = repo_path if repo_path.exists() else co_located_path
        if not path.exists():
            raise FileNotFoundError(
                f"N-gram file not found. Checked: {repo_path} and {co_located_path}. "
                "Run idea/featurizer/generate_features.py."
            )

        logger.info("  Loading n-gram features from %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            features = json.load(f)

        self.tag_patterns = {}
        for tag_name, tag_data in features["tags"].items():
            self.tag_patterns[tag_name] = {
                "unigrams": set(item["ngram"] for item in tag_data.get("unigrams", [])),
                "bigrams": set(item["ngram"] for item in tag_data.get("bigrams", [])),
                "trigrams": set(item["ngram"] for item in tag_data["trigrams"]),
                "fourgrams": set(item["ngram"] for item in tag_data["fourgrams"]),
                "fivegrams": set(item["ngram"] for item in tag_data.get("fivegrams", []))
            }

        self.tags_sorted = sorted(self.tag_patterns.keys())
        logger.info("    Loaded features for %d tags", len(self.tags_sorted))

    @staticmethod
    def _extract_ngram_words(text: str, n: int) -> set:
        words = text.split()
        if len(words) < n:
            return set()
        return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def _compute_pattern_features(self, question: str) -> np.ndarray:
        q_unigrams = self._extract_ngram_words(question, 1)
        q_bigrams = self._extract_ngram_words(question, 2)
        q_trigrams = self._extract_ngram_words(question, 3)
        q_fourgrams = self._extract_ngram_words(question, 4)
        q_fivegrams = self._extract_ngram_words(question, 5)

        feat = []
        for tag in self.tags_sorted:
            unigram_matches = len(q_unigrams & self.tag_patterns[tag]["unigrams"])
            bigram_matches = len(q_bigrams & self.tag_patterns[tag]["bigrams"])
            trigram_matches = len(q_trigrams & self.tag_patterns[tag]["trigrams"])
            fourgram_matches = len(q_fourgrams & self.tag_patterns[tag]["fourgrams"])
            fivegram_matches = len(q_fivegrams & self.tag_patterns[tag]["fivegrams"])
            feat.extend([unigram_matches, bigram_matches, trigram_matches, fourgram_matches, fivegram_matches])
        return np.array(feat, dtype=np.float32)

    def predict(
        self,
        question: str,
        top_k: int = 3,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if self.use_token_attention:
            # Token-level attention path
            # Apply E5-instruct prefix
            question_prefixed = E5PrefixConfig.format_classifier_query(question)

            # Extract token embeddings (shape: [1, seq_len, token_dim])
            token_embeddings, attention_mask = extract_token_embeddings(
                model=self.embedding_model,
                texts=[question_prefixed],
                max_length=512,
                normalize=True,
                batch_size=1,
                show_progress=False
            )

            # Convert to tensors
            token_tensor = token_embeddings.to(self.device)
            mask_tensor = attention_mask.to(self.device)

            with torch.no_grad():
                logits = self.model(token_tensor, mask_tensor)
                probs = F.softmax(logits, dim=1)[0]
                top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                top_probs = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

        else:
            # Legacy pooled embedding + n-gram patterns path
            # Use pre-computed embedding if provided, otherwise encode
            if precomputed_embedding is not None:
                embedding = precomputed_embedding
            else:
                # Apply E5-instruct prefix
                question_prefixed = E5PrefixConfig.format_classifier_query(question)

                embedding = self.embedding_model.encode(
                    [question_prefixed],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]

            pattern = self._compute_pattern_features(question)
            pattern = (pattern - self.pattern_mean) / self.pattern_std

            emb_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            pat_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(emb_tensor, pat_tensor)
                probs = F.softmax(logits, dim=1)[0]
                top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                top_probs = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            tag = self.tag_encoder[idx]
            predictions.append({
                'tag': tag,
                'confidence': float(prob),
                'answer': self.tag_to_answer.get(tag, "Answer not found")
            })

        return {
            'results': predictions,
            'metadata': {
                'num_candidates': len(predictions)
            }
        }

    def search(
        self,
        query: str,
        top_k: int,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        start = time.time()
        response = self.predict(query, top_k, precomputed_embedding=precomputed_embedding)
        response['metadata']['inference_time_ms'] = round((time.time() - start) * 1000, 2)
        return response
