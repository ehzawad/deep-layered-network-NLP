"""
CLI shim for training / fine-tuning the 203-tag classifier used by idea.pipeline.

The heavy lifting is delegated to the UnifiedTagClassifierTrainer defined inside
idea/training/unified_tag_classifier_trainer.py, keeping all plumbing local to
the idea package.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import types

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.training.unified_tag_classifier_trainer import UnifiedTagClassifierTrainer
from idea.utils.cache import compute_fingerprint
from idea.config import E5PrefixConfig, DEFAULT_EMBEDDING_MODEL

LOG = logging.getLogger("idea.training.tag_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 203-tag classifier used by idea.pipeline with FiLM conditioning and auxiliary losses"
    )
    parser.add_argument("--models", default="idea/models/tag_classifier", help="Output directory for checkpoints")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer backbone")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (trainer uses early stopping)")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size (default: 48, reduced for larger model)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002, reduced for deeper model)")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument("--dropout", type=float, default=0.55, help="Dropout probability (default: 0.55)")
    parser.add_argument("--ngram-mode", choices=("manual", "auto"), default="manual", help="Which featurizer output to consume")
    parser.add_argument("--use-film", action="store_true", default=True, help="Enable FiLM conditioning (default: True)")
    parser.add_argument("--no-film", action="store_false", dest="use_film", help="Disable FiLM conditioning")
    parser.add_argument("--use-auxiliary", action="store_true", default=True, help="Enable auxiliary classification heads (default: True)")
    parser.add_argument("--no-auxiliary", action="store_false", dest="use_auxiliary", help="Disable auxiliary heads")
    parser.add_argument("--aux-loss-weight", type=float, default=0.3, help="Weight for auxiliary losses (default: 0.3)")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use Focal Loss (default: True)")
    parser.add_argument("--no-focal-loss", action="store_false", dest="use_focal_loss", help="Use CrossEntropy instead of Focal Loss")
    parser.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss alpha parameter (default: 1.0)")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter (default: 2.0)")
    parser.add_argument("--force", action="store_true", help="Ignore cache and retrain")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    models_dir = Path(args.models)
    if not models_dir.is_absolute():
        models_dir = ROOT_DIR / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Output directory: %s", models_dir.resolve())
    LOG.info("Embedding model: %s", args.embedding_model)
    LOG.info("Using %s n-gram features", args.ngram_mode)

    trainer = UnifiedTagClassifierTrainer(embedding_model=args.embedding_model)

    # Override the featurizer selection without editing the upstream trainer.
    original_loader = trainer.load_ngram_features

    def load_with_custom_mode(self, active_ngram_file="manual"):
        return original_loader(args.ngram_mode)

    trainer.load_ngram_features = types.MethodType(load_with_custom_mode, trainer)

    df_train, df_eval = trainer.load_data()
    datasets_dir = Path(__file__).parent.parent / "datasets"
    features_file = datasets_dir / "features" / f"{args.ngram_mode}_ngrams.json"
    metadata_file = models_dir / "unified_tag_classifier_metadata.json"
    model_file = models_dir / "unified_tag_classifier.pth"

    cache_inputs = [
        datasets_dir / "sts_train.csv",
        datasets_dir / "sts_eval.csv",
        features_file,
    ]
    cache_extra = json.dumps({
        'embedding_model': args.embedding_model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'dropout': args.dropout,
        'ngram_mode': args.ngram_mode,
        'use_film': args.use_film,
        'use_auxiliary': args.use_auxiliary,
        'aux_loss_weight': args.aux_loss_weight,
        'use_focal_loss': args.use_focal_loss,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'e5_prefix_config': E5PrefixConfig.get_cache_key()
    }, sort_keys=True)
    fingerprint = compute_fingerprint(cache_inputs, cache_extra)

    if not args.force and metadata_file.exists() and model_file.exists():
        try:
            with metadata_file.open('r', encoding='utf-8') as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = {}
        if existing.get('fingerprint') == fingerprint:
            LOG.info("No input changes detected; reusing cached classifier. Use --force to retrain.")
            return
    LOG.info("Training rows: %d | Eval rows: %d | Tags: %d", len(df_train), len(df_eval), df_train['tag'].nunique())

    start_time = time.time()
    model = trainer.train(
        df_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dropout=args.dropout,
        use_film=args.use_film,
        use_auxiliary=args.use_auxiliary,
        aux_loss_weight=args.aux_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    training_minutes = (time.time() - start_time) / 60

    trainer.save_model(model, models_dir, fingerprint=fingerprint)
    LOG.info("Training finished in %.2f minutes", training_minutes)


if __name__ == "__main__":
    main()
