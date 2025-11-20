"""Utility to verify that exact match questions exist in STS train/eval datasets."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Set


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExactMatchEntry:
    question: str
    tag_train: str
    tag_eval: str


def load_exact_matches(path: Path) -> list[ExactMatchEntry]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"question", "tag_train", "tag_eval"}
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns {sorted(missing)} in {path}"
            )
        entries: list[ExactMatchEntry] = []
        for row in reader:
            entries.append(
                ExactMatchEntry(
                    question=row["question"],
                    tag_train=row["tag_train"],
                    tag_eval=row["tag_eval"],
                )
            )
    return entries


def load_question_tag_index(path: Path) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"question", "tag"}
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns {sorted(missing)} in {path}"
            )
        for row in reader:
            index[row["question"]].add(row["tag"])
    return index


def verify_exact_matches(
    exact_entries: Iterable[ExactMatchEntry],
    train_index: Dict[str, Set[str]],
    eval_index: Dict[str, Set[str]],
) -> dict[str, list[str]]:
    report = {
        "train_missing": [],
        "train_tag_mismatch": [],
        "eval_missing": [],
        "eval_tag_mismatch": [],
    }

    for entry in exact_entries:
        train_tags = train_index.get(entry.question)
        if not train_tags:
            report["train_missing"].append(entry.question)
        elif entry.tag_train not in train_tags:
            report["train_tag_mismatch"].append(
                f"{entry.question} (expected tag {entry.tag_train}, found {sorted(train_tags)})"
            )

        eval_tags = eval_index.get(entry.question)
        if not eval_tags:
            report["eval_missing"].append(entry.question)
        elif entry.tag_eval not in eval_tags:
            report["eval_tag_mismatch"].append(
                f"{entry.question} (expected tag {entry.tag_eval}, found {sorted(eval_tags)})"
            )

    return report


def format_report(report: dict[str, list[str]], total: int) -> str:
    lines = [f"Checked {total} exact-match questions."]
    issues = sum(len(values) for values in report.values())
    if issues == 0:
        lines.append("All questions were located with matching tags in both datasets.")
        return "\n".join(lines)

    lines.append("Found mismatches:")
    for key, entries in report.items():
        if entries:
            lines.append(f"- {key}: {len(entries)}")
            lines.extend(f"    * {entry}" for entry in entries)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact",
        type=Path,
        default=BASE_DIR / "exact_same.csv",
        help="Path to the exact match CSV (default: datasets/exact_same.csv)",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=BASE_DIR / "sts_train.csv",
        help="Path to the STS train CSV (default: datasets/sts_train.csv)",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=BASE_DIR / "sts_eval.csv",
        help="Path to the STS eval CSV (default: datasets/sts_eval.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_entries = load_exact_matches(args.exact)
    train_index = load_question_tag_index(args.train)
    eval_index = load_question_tag_index(args.eval)

    report = verify_exact_matches(exact_entries, train_index, eval_index)
    output = format_report(report, len(exact_entries))
    print(output)

    has_issue = any(report.values())
    return 1 if has_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
