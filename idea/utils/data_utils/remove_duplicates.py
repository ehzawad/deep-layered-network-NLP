#!/usr/bin/env python
"""
Remove duplicate questions from STS datasets (keeps first occurrence).

Usage:
    python idea/datasets/remove_duplicates.py
"""

import csv
from pathlib import Path
from collections import OrderedDict


def remove_duplicates(input_path: Path, output_path: Path):
    """Remove duplicate questions, keeping first occurrence."""

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    # Read all rows
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if "question" not in fieldnames:
            print(f"Error: 'question' column not found in {input_path}")
            return

        # Use OrderedDict to preserve order while removing duplicates
        # Key: question text, Value: full row dict
        unique_rows = OrderedDict()
        duplicates_found = 0

        for row in reader:
            question = row["question"]
            if question not in unique_rows:
                unique_rows[question] = row
            else:
                duplicates_found += 1

    # Write deduplicated rows
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows.values())

    original_count = len(unique_rows) + duplicates_found
    final_count = len(unique_rows)

    print(f"✓ {input_path.name}")
    print(f"  Original rows: {original_count}")
    print(f"  Duplicates removed: {duplicates_found}")
    print(f"  Final rows: {final_count}")
    print(f"  Saved to: {output_path}")
    print()


def main():
    datasets_dir = Path(__file__).parent

    files_to_clean = [
        ("sts_eval.csv", "sts_eval_dedup.csv"),
        ("sts_train.csv", "sts_train_dedup.csv")
    ]

    print("=" * 60)
    print("REMOVING DUPLICATE QUESTIONS FROM STS DATASETS")
    print("=" * 60)
    print()

    for input_name, output_name in files_to_clean:
        input_path = datasets_dir / input_name
        output_path = datasets_dir / output_name
        remove_duplicates(input_path, output_path)

    print("=" * 60)
    print("DEDUPLICATION COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review the deduplicated files:")
    print("   - sts_eval_dedup.csv")
    print("   - sts_train_dedup.csv")
    print()
    print("2. If satisfied, replace original files:")
    print("   mv idea/datasets/sts_eval_dedup.csv idea/datasets/sts_eval.csv")
    print("   mv idea/datasets/sts_train_dedup.csv idea/datasets/sts_train.csv")
    print()
    print("3. Rebuild classifier and FAISS index with clean data")


if __name__ == "__main__":
    main()
