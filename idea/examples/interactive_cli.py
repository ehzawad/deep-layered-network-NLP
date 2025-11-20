#!/usr/bin/env python3
"""
Simple interactive CLI to probe pipelineNLP (IdeaPipeline) predictions.

Usage:
    python idea/examples/interactive_cli.py [--top-k 2]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea import IdeaPipeline
from idea.utils.artifacts import ensure_all

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive pipelineNLP probe.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many fused candidates to keep (set 2 to mirror production top-2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_all()
    pipe = IdeaPipeline()
    pipe.initialize()
    fusion_top_k = max(1, args.top_k)
    print(f"Interactive pipelineNLP (top-{fusion_top_k}; type 'quit' to exit)")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = pipe.run(question, fusion_top_k=fusion_top_k)
        tag = result.get("primary_tag")
        print(f"Top tag: {tag}")
        print("Candidates:")
        for idx, cand in enumerate(result.get("candidates", []), 1):
            print(f"  {idx}. {cand['tag']} (score={cand['final_score']:.3f})")
        print()

if __name__ == "__main__":
    sys.exit(main())
