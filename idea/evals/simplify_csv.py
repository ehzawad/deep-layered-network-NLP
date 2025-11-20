#!/usr/bin/env python
"""
Extract only the 5 essential columns from detailed evaluation CSV.

Usage:
    python idea/evals/simplify_csv.py detailed_eval_nov17.csv simple_eval_nov17.csv
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 5 essential columns from detailed eval CSV"
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV file (detailed evaluation results)"
    )
    parser.add_argument(
        "output_csv",
        help="Output CSV file (simplified, 5 columns only)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Required columns to extract
    required_columns = [
        "input_question",
        "mapped_question",
        "expected_tag",
        "predicted_tag",
        "cosine_similarity",
        "classifier_confidence",
        "prediction_source"
    ]

    # Read input CSV and extract only required columns
    simplified_rows = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Verify all required columns exist
        missing_cols = [col for col in required_columns if col not in reader.fieldnames]
        if missing_cols:
            print(f"Error: Missing columns in input CSV: {missing_cols}")
            return

        for row in reader:
            simplified_row = {col: row[col] for col in required_columns}
            simplified_rows.append(simplified_row)

    # Write simplified CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=required_columns)
        writer.writeheader()
        writer.writerows(simplified_rows)

    print(f"✓ Simplified CSV written to {output_path}")
    print(f"  Rows: {len(simplified_rows)}")
    print(f"  Columns: {', '.join(required_columns)}")


if __name__ == "__main__":
    main()
