#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STS N-gram Feature Generation CLI

Complete workflow for generating, analyzing, and cleaning n-gram features for 203 STS tags.

Workflow:
1. Extract top-K n-grams from sts_train.csv (203 tags)
2. Analyze overlaps across tags
3. Optionally clean shared n-grams using dominance-based filtering
4. Save to auto_ngrams.json, manual_ngrams.json, overlap_analysis.json, cleanup_report.json

Usage:
    # Extract and analyze (no cleanup)
    python idea/featurizer/generate_features.py --top-k 20

    # Extract, analyze, and auto-clean
    python idea/featurizer/generate_features.py --top-k 20 --auto-clean

    # Custom dominance ratio
    python idea/featurizer/generate_features.py --top-k 20 --auto-clean --dominance-ratio 2.5

    # Force overwrite manual file
    python idea/featurizer/generate_features.py --top-k 20 --auto-clean --force-overwrite
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path so `idea` package imports resolve
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from idea.featurizer import FEATURES_DIR
from idea.featurizer.ngram_extractor import STSNgramExtractor
from idea.featurizer.feature_analyzer import STSFeatureAnalyzer
from idea.featurizer.clean_ngrams import STSNgramCleaner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate STS n-gram features with optional overlap cleanup"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top n-grams to extract per tag (default: 40)"
    )
    parser.add_argument(
        "--auto-clean",
        action="store_true",
        help="Automatically clean shared n-grams using dominance filtering"
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=2.0,
        help="Minimum frequency ratio for dominance (default: 2.0)"
    )
    parser.add_argument(
        "--min-ngrams",
        type=int,
        default=5,
        help="Minimum n-grams to keep per tag (safety net, default: 5)"
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite manual_ngrams.json even if it exists"
    )
    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()

    print("="*70)
    print("STS N-GRAM FEATURE GENERATION")
    print("="*70)
    print(f"Top-K per tag: {args.top_k}")
    print(f"Auto-clean: {args.auto_clean}")
    if args.auto_clean:
        print(f"Dominance ratio: {args.dominance_ratio}x")
        print(f"Min n-grams per tag: {args.min_ngrams}")
    print(f"Force overwrite manual: {args.force_overwrite}")
    print("="*70)

    # Define paths
    sts_datasets_dir = Path(__file__).parent.parent / "datasets"
    train_csv_path = sts_datasets_dir / "sts_train.csv"

    auto_ngrams_path = FEATURES_DIR / "auto_ngrams.json"
    manual_ngrams_path = FEATURES_DIR / "manual_ngrams.json"
    overlap_analysis_path = FEATURES_DIR / "overlap_analysis.json"
    cleanup_report_path = FEATURES_DIR / "cleanup_report.json"

    # Validate train CSV exists
    if not train_csv_path.exists():
        print(f"\n❌ ERROR: Training data not found at {train_csv_path}")
        print(f"   Available files in {sts_datasets_dir}:")
        print(f"   {list(sts_datasets_dir.glob('*.csv'))}")
        sys.exit(1)

    try:
        # ===================================================================
        # STEP 1: Extract n-grams
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 1: EXTRACTING N-GRAMS")
        print("="*70)

        extractor = STSNgramExtractor(
            train_csv_path=train_csv_path,
            top_k=args.top_k
        )

        # Generate and save to auto file
        extractor.generate_and_save(
            auto_path=auto_ngrams_path,
            manual_path=manual_ngrams_path,
            overwrite_manual=args.force_overwrite
        )

        print(f"\n✅ Step 1 complete!")
        print(f"   Auto file: {auto_ngrams_path}")
        print(f"   Manual file: {manual_ngrams_path}")

        # ===================================================================
        # STEP 2: Analyze overlaps
        # ===================================================================
        print("\n" + "="*70)
        print("STEP 2: ANALYZING WITHIN-CLUSTER OVERLAPS")
        print("="*70)

        analyzer = STSFeatureAnalyzer(features_path=auto_ngrams_path)
        analysis = analyzer.analyze_and_save(output_path=overlap_analysis_path)

        print(f"\n✅ Step 2 complete!")
        print(f"   Overlap analysis: {overlap_analysis_path}")

        # ===================================================================
        # STEP 3: Clean shared n-grams (optional)
        # ===================================================================
        if args.auto_clean:
            print("\n" + "="*70)
            print("STEP 3: CLEANING SHARED N-GRAMS (WITHIN-CLUSTER)")
            print("="*70)

            cleaner = STSNgramCleaner(
                auto_ngrams_path=auto_ngrams_path,
                overlap_analysis_path=overlap_analysis_path,
                dominance_ratio=args.dominance_ratio,
                min_ngrams_per_tag=args.min_ngrams
            )

            cleaned_features, report = cleaner.clean_and_save(
                manual_ngrams_path=manual_ngrams_path,
                cleanup_report_path=cleanup_report_path
            )

            print(f"\n✅ Step 3 complete!")
            print(f"   Cleaned manual file: {manual_ngrams_path}")
            print(f"   Cleanup report: {cleanup_report_path}")

        else:
            print("\n⏭️  Step 3: Skipped (use --auto-clean to enable)")

        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print("\n" + "="*70)
        print("GENERATION COMPLETE!")
        print("="*70)

        print(f"\n📁 Generated files:")
        print(f"   1. {auto_ngrams_path}")
        print(f"      → Auto-generated features (always regenerated)")
        print(f"   2. {manual_ngrams_path}")
        print(f"      → Manual features (edit this for customization)")
        print(f"   3. {overlap_analysis_path}")
        print(f"      → Overlap analysis")

        if args.auto_clean:
            print(f"   4. {cleanup_report_path}")
            print(f"      → Cleanup report (dominance-based filtering)")

        print(f"\n📊 Feature summary:")
        print(f"   Tags: {len(extractor.tags_sorted)}")
        print(f"   Top-K per tag: {args.top_k}")
        print(f"   Total n-grams: {len(extractor.tags_sorted) * args.top_k * 5}")
        print(f"   Pattern dimension: {len(extractor.tags_sorted) * 5} (5 features × {len(extractor.tags_sorted)} tags)")

        if args.auto_clean:
            summary = report["summary"]
            print(f"\n🧹 Cleanup results:")
            print(f"   Trigrams removed: {summary['trigrams']['removed']}/{summary['trigrams']['original']} "
                  f"({summary['trigrams']['removed_pct']:.1f}%)")
            print(f"   Fourgrams removed: {summary['fourgrams']['removed']}/{summary['fourgrams']['original']} "
                  f"({summary['fourgrams']['removed_pct']:.1f}%)")

        print(f"\n💡 Next steps:")
        print(f"   1. Review {manual_ngrams_path} and customize if needed")
        print(f"   2. Update STS config to use 'manual' or 'auto' features")
        print(f"   3. Modify UnifiedTagClassifier architecture (pattern_dim = 3 × num_tags)")
        print(f"   4. Update training script to load features from JSON")
        print(f"   5. Train new model with per-tag n-gram features")

        print("\n" + "="*70)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR during feature generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
