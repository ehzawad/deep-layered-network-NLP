#!/usr/bin/env python
"""
Analyze confusion matrix from evaluation results to identify which tags get confused.

Usage:
    python idea/evals/eval_confusion_matrix.py \
        --eval-csv idea/evals/results/nov-18-ec.csv \
        --output-dir idea/evals/results/confusion_analysis/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confusion matrix from evaluation results")
    parser.add_argument(
        "--eval-csv",
        required=True,
        help="Path to evaluation CSV (output from eval_pipeline.py)"
    )
    parser.add_argument(
        "--output-dir",
        default="idea/evals/results/confusion_analysis",
        help="Directory to write analysis outputs"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top confused pairs to show (default: 20)"
    )
    return parser.parse_args()


def load_eval_results(csv_path: Path) -> list[dict]:
    """Load evaluation results from CSV."""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_confusion_matrix(rows: list[dict]) -> dict:
    """
    Build confusion matrix from evaluation results.

    Returns:
        confusion[expected_tag][predicted_tag] = count
    """
    confusion = defaultdict(lambda: defaultdict(int))

    for row in rows:
        expected = row.get("expected_tag", "")
        predicted = row.get("predicted_tag", "")

        if expected and predicted:
            confusion[expected][predicted] += 1

    return confusion


def calculate_per_tag_accuracy(confusion: dict) -> dict:
    """Calculate per-tag accuracy from confusion matrix."""
    per_tag = {}

    for expected_tag in confusion:
        total = sum(confusion[expected_tag].values())
        correct = confusion[expected_tag].get(expected_tag, 0)
        accuracy = correct / total if total > 0 else 0.0

        per_tag[expected_tag] = {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy
        }

    return per_tag


def find_top_confusions(confusion: dict, top_n: int = 20) -> list[tuple]:
    """
    Find the most common confusion pairs (excluding correct predictions).

    Returns:
        List of (expected_tag, predicted_tag, count, bidirectional_count)
    """
    confusions = []

    for expected in confusion:
        for predicted in confusion[expected]:
            if expected != predicted:  # Exclude correct predictions
                count = confusion[expected][predicted]
                # Check reverse confusion for symmetry analysis
                reverse_count = confusion[predicted].get(expected, 0)
                bidirectional = count + reverse_count

                confusions.append((expected, predicted, count, reverse_count, bidirectional))

    # Sort by bidirectional count (most confused pairs)
    confusions.sort(key=lambda x: x[4], reverse=True)

    return confusions[:top_n]


def find_example_questions(rows: list[dict], expected_tag: str, predicted_tag: str, limit: int = 3) -> list[str]:
    """Find example questions for a specific confusion pair."""
    examples = []

    for row in rows:
        if row.get("expected_tag") == expected_tag and row.get("predicted_tag") == predicted_tag:
            question = row.get("input_question", "") or row.get("question", "")
            if question:
                examples.append(question)

            if len(examples) >= limit:
                break

    return examples


def generate_text_report(
    confusion: dict,
    per_tag: dict,
    top_confusions: list,
    rows: list[dict],
    output_dir: Path
):
    """Generate human-readable text report."""
    report_path = output_dir / "confusion_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        total_samples = sum(stats["total"] for stats in per_tag.values())
        total_correct = sum(stats["correct"] for stats in per_tag.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Overall accuracy: {overall_accuracy * 100:.2f}%\n")
        f.write(f"Total tags: {len(per_tag)}\n\n")

        # Top confused pairs
        f.write("=" * 80 + "\n")
        f.write(f"TOP {len(top_confusions)} MOST CONFUSED TAG PAIRS\n")
        f.write("=" * 80 + "\n\n")

        for i, (expected, predicted, count_ep, count_pe, bidirectional) in enumerate(top_confusions, 1):
            f.write(f"{i}. {expected} ⇄ {predicted}\n")
            f.write(f"   Expected→Predicted: {count_ep} errors\n")
            f.write(f"   Predicted→Expected: {count_pe} errors\n")
            f.write(f"   Total bidirectional: {bidirectional} errors\n")

            # Show if confusion is symmetric or asymmetric
            if count_ep > 0 and count_pe > 0:
                ratio = max(count_ep, count_pe) / min(count_ep, count_pe)
                if ratio < 2:
                    f.write(f"   Pattern: SYMMETRIC (ratio: {ratio:.2f})\n")
                else:
                    f.write(f"   Pattern: ASYMMETRIC (ratio: {ratio:.2f})\n")
            else:
                f.write(f"   Pattern: ONE-WAY confusion\n")

            # Example questions
            examples = find_example_questions(rows, expected, predicted, limit=2)
            if examples:
                f.write(f"   Example questions (expected: {expected}):\n")
                for ex in examples:
                    f.write(f"     - {ex[:100]}...\n" if len(ex) > 100 else f"     - {ex}\n")

            f.write("\n")

        # Per-tag accuracy breakdown
        f.write("=" * 80 + "\n")
        f.write("PER-TAG ACCURACY (Lowest performing tags)\n")
        f.write("=" * 80 + "\n\n")

        sorted_tags = sorted(per_tag.items(), key=lambda x: x[1]["accuracy"])[:30]

        for tag, stats in sorted_tags:
            f.write(f"{tag}\n")
            f.write(f"  Accuracy: {stats['accuracy'] * 100:.2f}% ({stats['correct']}/{stats['total']})\n")
            f.write(f"  Errors: {stats['incorrect']}\n\n")

    print(f"✓ Text report written to: {report_path}")


def generate_csv_reports(
    confusion: dict,
    per_tag: dict,
    top_confusions: list,
    output_dir: Path
):
    """Generate CSV reports for further analysis."""

    # 1. Full confusion matrix (sparse format)
    confusion_csv = output_dir / "confusion_matrix.csv"
    with confusion_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expected_tag", "predicted_tag", "count"])

        for expected in sorted(confusion.keys()):
            for predicted in sorted(confusion[expected].keys()):
                count = confusion[expected][predicted]
                writer.writerow([expected, predicted, count])

    print(f"✓ Confusion matrix CSV written to: {confusion_csv}")

    # 2. Top confused pairs
    confusions_csv = output_dir / "top_confused_pairs.csv"
    with confusions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "expected_tag",
            "predicted_tag",
            "expected_to_predicted_count",
            "predicted_to_expected_count",
            "bidirectional_total",
            "pattern"
        ])

        for i, (expected, predicted, count_ep, count_pe, bidirectional) in enumerate(top_confusions, 1):
            if count_ep > 0 and count_pe > 0:
                ratio = max(count_ep, count_pe) / min(count_ep, count_pe)
                pattern = "symmetric" if ratio < 2 else "asymmetric"
            else:
                pattern = "one-way"

            writer.writerow([i, expected, predicted, count_ep, count_pe, bidirectional, pattern])

    print(f"✓ Top confused pairs CSV written to: {confusions_csv}")

    # 3. Per-tag accuracy
    per_tag_csv = output_dir / "per_tag_accuracy.csv"
    with per_tag_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "total_samples", "correct", "incorrect", "accuracy"])

        for tag in sorted(per_tag.keys(), key=lambda t: per_tag[t]["accuracy"]):
            stats = per_tag[tag]
            writer.writerow([
                tag,
                stats["total"],
                stats["correct"],
                stats["incorrect"],
                f"{stats['accuracy']:.4f}"
            ])

    print(f"✓ Per-tag accuracy CSV written to: {per_tag_csv}")


def main():
    args = parse_args()

    # Resolve paths
    csv_path = Path(args.eval_csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluation results from: {csv_path}")
    rows = load_eval_results(csv_path)
    print(f"✓ Loaded {len(rows)} evaluation samples")

    print("\nBuilding confusion matrix...")
    confusion = build_confusion_matrix(rows)
    print(f"✓ Found {len(confusion)} unique tags")

    print("\nCalculating per-tag accuracy...")
    per_tag = calculate_per_tag_accuracy(confusion)

    print(f"\nFinding top {args.top_n} confused pairs...")
    top_confusions = find_top_confusions(confusion, top_n=args.top_n)

    print("\nGenerating reports...")
    generate_text_report(confusion, per_tag, top_confusions, rows, output_dir)
    generate_csv_reports(confusion, per_tag, top_confusions, output_dir)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print("  - confusion_report.txt        (Human-readable analysis)")
    print("  - confusion_matrix.csv        (Full confusion matrix)")
    print("  - top_confused_pairs.csv      (Top confused tag pairs)")
    print("  - per_tag_accuracy.csv        (Per-tag performance)")


if __name__ == "__main__":
    main()
