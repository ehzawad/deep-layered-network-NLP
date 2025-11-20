#!/usr/bin/env python3
"""
Test script for token attention classifier.

This script demonstrates:
1. Training a token attention model
2. Running inference with token attention
3. Comparing results with the legacy unified classifier

Usage:
    # Train lightweight token attention model
    python idea/examples/test_token_attention.py --train

    # Test inference only (requires pre-trained model)
    python idea/examples/test_token_attention.py

    # Train and test
    python idea/examples/test_token_attention.py --train --test
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea.config import TagClassifierConfig
from idea.inference.tag_classifier_matcher import TagClassifierMatcher
from idea.training.token_attention_trainer import TokenAttentionTrainer


def train_token_attention(variant="lightweight", force=False):
    """Train token attention model"""
    print("\n" + "="*60)
    print(f"TRAINING TOKEN ATTENTION MODEL ({variant})")
    print("="*60 + "\n")

    trainer = TokenAttentionTrainer(variant=variant)
    trainer.train_model(
        output_dir="idea/models/tag_classifier",
        epochs=50,
        batch_size=32,
        lr=0.0001,
        dropout=0.5,
        patience=10,
        use_focal_loss=True,
        focal_alpha=1.0,
        focal_gamma=2.0,
        hidden_dim=256,  # For lightweight
        num_heads=8,     # For multihead
        force=force
    )


def test_inference(use_token_attention=False, variant="lightweight"):
    """Test inference with a few sample questions"""
    print("\n" + "="*60)
    if use_token_attention:
        print(f"TESTING TOKEN ATTENTION INFERENCE ({variant})")
    else:
        print("TESTING LEGACY UNIFIED CLASSIFIER INFERENCE")
    print("="*60 + "\n")

    # Sample test questions (Bangla)
    test_questions = [
        "আমার স্বামীর নাম ঠিক করতে কি লাগবে?",
        "আব্বার নাম ঠিক করতে কি লাগবে?",
        "আমার মায়ের নাম ঠিক করতে চাই",
        "জন্মতারিখ পরিবর্তন করতে কি করতে হবে?",
    ]

    # Initialize matcher
    config = TagClassifierConfig(
        use_token_attention=use_token_attention,
        token_attention_variant=variant
    )
    matcher = TagClassifierMatcher(config)
    matcher.initialize()

    # Run inference on test questions
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = matcher.predict(question, top_k=3)

        print("   Predictions:")
        for j, pred in enumerate(result['results'], 1):
            print(f"      {j}. {pred['tag']} (confidence: {pred['confidence']:.4f})")


def compare_models():
    """Compare legacy vs token attention side-by-side"""
    print("\n" + "="*60)
    print("COMPARING LEGACY vs TOKEN ATTENTION")
    print("="*60 + "\n")

    test_questions = [
        "আমার স্বামীর নাম ঠিক করতে কি লাগবে?",
        "আব্বার নাম ঠিক করতে কি লাগবে?",
    ]

    # Initialize both matchers
    legacy_config = TagClassifierConfig(use_token_attention=False)
    token_config = TagClassifierConfig(use_token_attention=True, token_attention_variant="lightweight")

    legacy_matcher = TagClassifierMatcher(legacy_config)
    token_matcher = TagClassifierMatcher(token_config)

    print("Initializing legacy matcher...")
    legacy_matcher.initialize()
    print("\nInitializing token attention matcher...")
    token_matcher.initialize()

    # Compare predictions
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}")

        # Legacy predictions
        legacy_result = legacy_matcher.predict(question, top_k=3)
        print("\n[LEGACY UNIFIED CLASSIFIER]")
        for j, pred in enumerate(legacy_result['results'], 1):
            print(f"  {j}. {pred['tag']} ({pred['confidence']:.4f})")

        # Token attention predictions
        token_result = token_matcher.predict(question, top_k=3)
        print("\n[TOKEN ATTENTION]")
        for j, pred in enumerate(token_result['results'], 1):
            print(f"  {j}. {pred['tag']} ({pred['confidence']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Test token attention classifier")
    parser.add_argument("--train", action="store_true", help="Train token attention model")
    parser.add_argument("--test", action="store_true", help="Test inference")
    parser.add_argument("--compare", action="store_true", help="Compare legacy vs token attention")
    parser.add_argument("--variant", type=str, default="lightweight",
                        choices=["lightweight", "multihead"],
                        help="Model variant to use")
    parser.add_argument("--force", action="store_true", help="Force retrain")

    args = parser.parse_args()

    # Default: if no flags, run comparison (assumes both models exist)
    if not (args.train or args.test or args.compare):
        args.compare = True

    try:
        if args.train:
            train_token_attention(variant=args.variant, force=args.force)

        if args.test:
            test_inference(use_token_attention=True, variant=args.variant)

        if args.compare:
            compare_models()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nHint: You may need to train the model first:")
        print(f"  python {sys.argv[0]} --train --variant {args.variant}")
        return 1

    print("\n✓ Done!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
