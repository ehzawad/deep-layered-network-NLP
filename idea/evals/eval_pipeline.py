#!/usr/bin/env python
"""
Evaluate pipelineNLP (IdeaPipeline) on a labeled CSV (question, tag).

Usage:
    python idea/evals/eval_pipeline.py --data idea/datasets/sts_eval.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea import IdeaPipeline
from idea.utils.artifacts import ensure_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipelineNLP accuracy on a CSV file.")
    parser.add_argument(
        "--data",
        default="idea/datasets/sts_eval.csv",
        help="CSV file with columns: question, tag (default: idea/datasets/sts_eval.csv)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many fused candidates to consider for top-k accuracy (default: 1)"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to emit per-row evaluation details as CSV (default: disabled)"
    )
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def main():
    args = parse_args()
    data_path = _resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {data_path}")

    # Load eval rows
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question" not in reader.fieldnames or "tag" not in reader.fieldnames:
            raise ValueError("CSV must contain 'question' and 'tag' columns")
        for line in reader:
            rows.append({"question": line["question"], "tag": line["tag"]})

    if not rows:
        print("No rows found in eval CSV.")
        return

    # Initialize pipeline
    ensure_all()
    pipeline = IdeaPipeline()
    pipeline.initialize()

    total = len(rows)
    top1_correct = 0
    topk_correct = 0
    misses_top1 = []
    misses_topk = []

    report_rows = []

    for idx, row in enumerate(rows, 1):
        question = row["question"]
        expected_tag = row["tag"]
        result = pipeline.run(question, fusion_top_k=args.top_k)
        candidates = result.get("candidates", [])
        dropped_candidates = result.get("dropped_candidates", [])
        predicted_tag = result.get("primary_tag")

        # Extract mapped question, cosine similarity, classifier confidence, and source from top candidate
        mapped_question = ""
        cosine_similarity = None
        classifier_confidence = None
        prediction_source = ""
        if candidates:
            best_candidate = candidates[0]
            prediction_source = best_candidate.get("source", "")
            signals = best_candidate.get("signals", {})

            # Extract STS signal
            sts_signal = signals.get("sts", {})
            if sts_signal:
                mapped_question = sts_signal.get("question", "")
                cosine_similarity = sts_signal.get("raw_score") or sts_signal.get("similarity")

            # Extract classifier signal
            clf_signal = signals.get("classifier", {})
            if clf_signal:
                classifier_confidence = clf_signal.get("raw_confidence") or clf_signal.get("confidence")

        if predicted_tag == expected_tag:
            top1_correct += 1
        else:
            misses_top1.append((question, expected_tag, predicted_tag))

        candidate_tags = [cand["tag"] for cand in candidates[:args.top_k]]
        if expected_tag in candidate_tags:
            topk_correct += 1
        else:
            misses_topk.append((question, expected_tag, candidate_tags))

        top_k_scores = [
            f"{cand.get('final_score', ''):.6f}" if cand.get("final_score") is not None else ""
            for cand in candidates[:args.top_k]
        ]
        top_k_sources = [cand.get("source", "") for cand in candidates[:args.top_k]]

        # Collect ALL fused scores (both above and below threshold)
        all_fused_candidates = candidates + dropped_candidates
        all_fused_tags = [cand["tag"] for cand in all_fused_candidates]
        all_fused_scores = [
            f"{cand.get('final_score', ''):.6f}" if cand.get("final_score") is not None else ""
            for cand in all_fused_candidates
        ]
        all_fused_sources = [cand.get("source", "") for cand in all_fused_candidates]

        report_rows.append(
            {
                # Primary analysis columns
                "input_question": question,
                "mapped_question": mapped_question,
                "expected_tag": expected_tag,
                "predicted_tag": predicted_tag or "",
                "cosine_similarity": f"{cosine_similarity:.6f}" if cosine_similarity is not None else "",
                "classifier_confidence": f"{classifier_confidence:.6f}" if classifier_confidence is not None else "",
                "prediction_source": prediction_source,
                # Additional details
                "question": question,
                "primary_tag": predicted_tag or "",
                "top1_correct": "1" if predicted_tag == expected_tag else "0",
                "topk_correct": "1" if expected_tag in candidate_tags else "0",
                "top_k_tags": "|".join(tag or "" for tag in candidate_tags),
                "top_k_scores": "|".join(top_k_scores),
                "top_k_sources": "|".join(top_k_sources),
                "all_fused_tags": "|".join(all_fused_tags),
                "all_fused_scores": "|".join(all_fused_scores),
                "all_fused_sources": "|".join(all_fused_sources),
                "num_dropped": str(len(dropped_candidates)),
            }
        )

        if idx % 100 == 0 or idx == total:
            current_top1_acc = 100.0 * top1_correct / idx
            current_topk_acc = 100.0 * topk_correct / idx
            print(f"Processed {idx}/{total} | Top-1: {current_top1_acc:.2f}% | Top-{args.top_k}: {current_topk_acc:.2f}%")

    top1_acc = 100.0 * top1_correct / total
    topk_acc = 100.0 * topk_correct / total

    print("\nEvaluation complete")
    print(f"Total rows: {total}")
    print(f"Top-1 accuracy: {top1_acc:.2f}%")
    print(f"Top-{args.top_k} accuracy: {topk_acc:.2f}%")
    print(f"Top-1 misses: {len(misses_top1)} (showing up to 5)")
    for q, exp, pred in misses_top1[:5]:
        print(f"- Expected {exp}, got {pred}; question snippet: {q[:60]}...")
    print(f"\nTop-{args.top_k} misses: {len(misses_topk)} (showing up to 5)")
    for q, exp, preds in misses_topk[:5]:
        print(f"- Expected {exp}, top-{args.top_k} candidates {preds}; snippet: {q[:60]}...")

    if args.output_csv:
        output_path = _resolve_path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                # Primary analysis columns (requested by user)
                "input_question",
                "mapped_question",
                "expected_tag",
                "predicted_tag",
                "cosine_similarity",
                "classifier_confidence",
                "prediction_source",
                # Additional details
                "question",
                "primary_tag",
                "top1_correct",
                "topk_correct",
                "top_k_tags",
                "top_k_scores",
                "top_k_sources",
                "all_fused_tags",
                "all_fused_scores",
                "all_fused_sources",
                "num_dropped",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nWrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
