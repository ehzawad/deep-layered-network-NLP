#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Token Attention Classifier

Entry point for training token-level attention classifiers that preserve positional
information for distinguishing similar tags.

Usage:
    # Train lightweight variant (recommended, ~2M params)
    python idea/training/train_token_attention.py --variant lightweight

    # Train multihead variant (heavier, ~8M params)
    python idea/training/train_token_attention.py --variant multihead

    # Custom hyperparameters
    python idea/training/train_token_attention.py \\
        --variant lightweight \\
        --epochs 100 \\
        --batch-size 48 \\
        --lr 0.0001 \\
        --dropout 0.5 \\
        --hidden-dim 256 \\
        --use-focal-loss \\
        --force
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.training.token_attention_trainer_efficient import TokenAttentionTrainer
from idea.config import DEFAULT_EMBEDDING_MODEL
from idea.utils.token_utils import TOKEN_MAX_LENGTH


def main():
    parser = argparse.ArgumentParser(
        description="Train token-level attention classifier for tag prediction"
    )

    # Model architecture
    parser.add_argument(
        "--variant",
        type=str,
        default="lightweight",
        choices=["lightweight", "multihead"],
        help="Model variant: lightweight (~2M params) or multihead (~8M params)"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="idea/models/tag_classifier",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability (default: 0.5)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )

    # Architecture-specific parameters
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for lightweight variant (default: 256)"
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads for multihead variant (default: 8)"
    )

    # Loss function
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use focal loss instead of cross-entropy"
    )

    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=1.0,
        help="Focal loss alpha parameter (default: 1.0)"
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (default: 2.0)"
    )

    # Control flags
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if model already exists"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=TOKEN_MAX_LENGTH,
        help=f"Tokenizer max_length for token attention (default: {TOKEN_MAX_LENGTH})"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable torch.autocast for faster training (fp16)."
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = TokenAttentionTrainer(
        embedding_model=args.embedding_model,
        variant=args.variant
    )

    # Train model
    trainer.train_model(
        output_dir=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        max_length=args.max_length,
        mixed_precision=args.mixed_precision,
        force=args.force
    )


if __name__ == "__main__":
    main()
