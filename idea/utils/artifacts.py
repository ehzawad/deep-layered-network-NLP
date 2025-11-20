"""Helpers that ensure required IDEA artifacts exist before runtime."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from idea.utils.cache import compute_fingerprint
from idea.config import E5PrefixConfig, DEFAULT_EMBEDDING_MODEL

LOG = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "idea" / "datasets" / "features"
SEMANTIC_DIR = ROOT / "idea" / "models" / "semantic"
CLASSIFIER_DIR = ROOT / "idea" / "models" / "tag_classifier"


def _run_step(args: list[str], desc: str):
    LOG.info("%s", desc)
    completed = subprocess.run(
        [sys.executable] + args,
        cwd=ROOT,
        env=os.environ,
        check=True
    )
    return completed


def ensure_features():
    """Ensure n-gram features exist and are up-to-date with training data."""
    manual = FEATURES_DIR / "manual_ngrams.json"
    auto = FEATURES_DIR / "auto_ngrams.json"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we need to regenerate features
    force_overwrite = False
    if manual.exists() and auto.exists():
        # Check if training data has changed (e.g., number of tags changed)
        datasets_dir = ROOT / "idea" / "datasets"
        train_file = datasets_dir / "sts_train.csv"

        if train_file.exists():
            try:
                # Load current training data to get actual tag count
                df_train = pd.read_csv(train_file)
                actual_num_tags = df_train['tag'].nunique()

                # Load metadata from existing features
                with manual.open('r', encoding='utf-8') as f:
                    features = json.load(f)
                    stored_num_tags = features.get('metadata', {}).get('num_tags', 0)

                if actual_num_tags != stored_num_tags:
                    LOG.warning(
                        "Training data changed: %d tags in CSV but features have %d tags. Regenerating features.",
                        actual_num_tags, stored_num_tags
                    )
                    force_overwrite = True
                else:
                    LOG.info("N-gram features are up-to-date (%d tags)", actual_num_tags)
                    return
            except (json.JSONDecodeError, KeyError, pd.errors.EmptyDataError) as e:
                LOG.warning("Could not verify feature freshness: %s. Regenerating.", e)
                force_overwrite = True
        else:
            LOG.info("Features exist, reusing them")
            return

    # Generate features
    args = [
        "idea/featurizer/generate_features.py",
        "--top-k",
        "40",
        "--auto-clean",
        "--dominance-ratio",
        "2.0",
    ]
    if force_overwrite:
        args.append("--force-overwrite")

    _run_step(args, "Generating STS n-gram features (auto + manual)")


def ensure_classifier():
    """Ensure tag classifier exists and is up-to-date with training data."""
    model_file = CLASSIFIER_DIR / "unified_tag_classifier.pth"
    metadata_file = CLASSIFIER_DIR / "unified_tag_classifier_metadata.json"
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we need to retrain the classifier
    force_rebuild = False
    if model_file.exists() and metadata_file.exists():
        # Compute current fingerprint
        datasets_dir = ROOT / "idea" / "datasets"
        cache_inputs = [
            datasets_dir / "sts_train.csv",
            datasets_dir / "sts_eval.csv",
            datasets_dir / "features" / "manual_ngrams.json",
        ]
        cache_extra = json.dumps({
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'epochs': 100,
            'batch_size': 48,          # Updated for FiLM model
            'lr': 0.0002,              # Updated for deeper model
            'patience': 15,
            'dropout': 0.55,           # New parameter
            'ngram_mode': 'manual',
            'use_film': True,          # New parameter
            'use_auxiliary': True,     # New parameter
            'aux_loss_weight': 0.3,    # New parameter
            'use_focal_loss': True,    # New parameter
            'focal_alpha': 1.0,        # New parameter
            'focal_gamma': 2.0,        # New parameter
            'e5_prefix_config': E5PrefixConfig.get_cache_key()
        }, sort_keys=True)

        try:
            current_fingerprint = compute_fingerprint(cache_inputs, cache_extra)

            # Load stored fingerprint
            with metadata_file.open('r', encoding='utf-8') as f:
                metadata = json.load(f)
                stored_fingerprint = metadata.get('fingerprint')

            if current_fingerprint != stored_fingerprint:
                LOG.warning(
                    "Classifier inputs changed (fingerprint mismatch). Retraining classifier."
                )
                force_rebuild = True
            else:
                LOG.info("Tag classifier is up-to-date")
                return
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            LOG.warning("Could not verify classifier freshness: %s. Retraining.", e)
            force_rebuild = True

    # Train classifier with FiLM conditioning
    args = [
        "idea/training/train_tag_classifier.py",
        "--models",
        str(CLASSIFIER_DIR),
        "--embedding-model",
        DEFAULT_EMBEDDING_MODEL,
        "--epochs",
        "100",
        "--batch-size",
        "48",              # Updated for FiLM model
        "--lr",
        "0.0002",          # Updated for deeper model
        "--dropout",
        "0.55",            # Updated for more regularization
        "--patience",
        "15",
        "--ngram-mode",
        "manual",
        "--use-film",      # Enable FiLM conditioning
        "--use-auxiliary", # Enable auxiliary losses
        "--use-focal-loss", # Enable focal loss
    ]
    if force_rebuild:
        args.append("--force")

    _run_step(args, "Training tag classifier (auto-trigger)")


def ensure_faiss_index():
    """Ensure FAISS index exists and is up-to-date with training data."""
    required = [
        SEMANTIC_DIR / "faiss_index_global.index",
        SEMANTIC_DIR / "sts_embeddings.npy",
        SEMANTIC_DIR / "question_mapping.csv",
    ]
    metadata_file = SEMANTIC_DIR / "sts_metadata.json"
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we need to rebuild the index
    force_rebuild = False
    if all(path.exists() for path in required) and metadata_file.exists():
        # Compute current fingerprint
        datasets_dir = ROOT / "idea" / "datasets"
        cache_inputs = [
            datasets_dir / "sts_train.csv",
            datasets_dir / "tag_to_answer.json",
        ]
        cache_extra = json.dumps({
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'normalize_embeddings': True,
            'indices_to_build': 'global',  # Must match build script (string, not list)
            'e5_prefix_config': E5PrefixConfig.get_cache_key()
        }, sort_keys=True)

        try:
            current_fingerprint = compute_fingerprint(cache_inputs, cache_extra)

            # Load stored fingerprint
            with metadata_file.open('r', encoding='utf-8') as f:
                metadata = json.load(f)
                stored_fingerprint = metadata.get('fingerprint')

            if current_fingerprint != stored_fingerprint:
                LOG.warning(
                    "FAISS index inputs changed (fingerprint mismatch). Rebuilding index."
                )
                force_rebuild = True
            else:
                LOG.info("FAISS semantic index is up-to-date")
                return
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            LOG.warning("Could not verify FAISS index freshness: %s. Rebuilding.", e)
            force_rebuild = True

    # Build index
    args = ["idea/training/build_faiss_indices.py", "--global"]
    if force_rebuild:
        args.append("--force")

    _run_step(args, "Building FAISS semantic index (auto-trigger)")


def ensure_all():
    ensure_features()
    ensure_classifier()
    ensure_faiss_index()

