"""
STS N-gram Feature Extractor

Generates n-gram features for all 203 tags:
- Top-K unigrams per tag
- Top-K bigrams per tag
- Top-K trigrams per tag
- Top-K fourgrams per tag
- Top-K fivegrams per tag
- Frequency and rank metadata

Different from classification: operates purely at the tag level.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


class STSNgramExtractor:
    """Extract and rank n-grams from STS training data"""

    def __init__(self, train_csv_path: Path, top_k: int = 40):
        """
        Initialize n-gram extractor

        Args:
            train_csv_path: Path to sts_train.csv
            top_k: Number of top n-grams to extract per tag (default: 40)
        """
        self.train_csv_path = train_csv_path
        self.top_k = top_k
        self.train_df = None
        self.tags_sorted = None

    def load_training_data(self) -> None:
        """Load STS training data from CSV"""
        print(f"Loading STS training data from {self.train_csv_path}")
        self.train_df = pd.read_csv(self.train_csv_path)

        print(f"Loaded {len(self.train_df)} samples")
        print(f"Columns: {list(self.train_df.columns)}")

        # Get unique tags
        self.tags_sorted = sorted(self.train_df['tag'].unique())
        print(f"Unique tags: {len(self.tags_sorted)}")

        print("✅ Training data loaded")

    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from text using sliding window

        Args:
            text: Input text
            n: N-gram size

        Returns:
            List of n-grams
        """
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def extract_tag_ngrams(self, tag: str, n: int) -> Counter:
        """
        Extract all n-grams for a specific tag

        Args:
            tag: Tag name
            n: N-gram size (3 for trigrams, 4 for fourgrams, 5 for fivegrams)

        Returns:
            Counter object with n-gram frequencies
        """
        tag_questions = self.train_df[self.train_df['tag'] == tag]['question']

        all_ngrams = []
        for question in tag_questions:
            all_ngrams.extend(self.extract_ngrams(question, n))

        return Counter(all_ngrams)

    def generate_ngram_features(self) -> Dict:
        """
        Generate n-gram features for all tags

        Returns:
            Dictionary with structure:
            {
                "metadata": {...},
                "tags": {
                    "tag_name": {
                        "trigrams": [{ngram, frequency, rank}, ...],
                        "fourgrams": [{ngram, frequency, rank}, ...],
                        "fivegrams": [{ngram, frequency, rank}, ...],
                    }
                }
            }
        """
        print("\nExtracting n-gram features for STS tags...")

        features = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "training_samples": len(self.train_df),
                "num_tags": len(self.tags_sorted),
                "top_k": self.top_k,
                "ngram_types": ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"],
                "pattern_dim": len(self.tags_sorted) * 5  # 5 features per tag
            },
            "tags": {}
        }

        for tag in self.tags_sorted:
            print(f"Processing tag: {tag}")

            # Extract unigrams
            unigram_counts = self.extract_tag_ngrams(tag, 1)
            top_unigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    unigram_counts.most_common(self.top_k)
                )
            ]

            # Extract bigrams
            bigram_counts = self.extract_tag_ngrams(tag, 2)
            top_bigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    bigram_counts.most_common(self.top_k)
                )
            ]

            # Extract trigrams
            trigram_counts = self.extract_tag_ngrams(tag, 3)
            top_trigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    trigram_counts.most_common(self.top_k)
                )
            ]

            # Extract fourgrams
            fourgram_counts = self.extract_tag_ngrams(tag, 4)
            top_fourgrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    fourgram_counts.most_common(self.top_k)
                )
            ]

            # Extract fivegrams
            fivegram_counts = self.extract_tag_ngrams(tag, 5)
            top_fivegrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    fivegram_counts.most_common(self.top_k)
                )
            ]

            features["tags"][tag] = {
                "group": "all_tags",
                "unigrams": top_unigrams,
                "bigrams": top_bigrams,
                "trigrams": top_trigrams,
                "fourgrams": top_fourgrams,
                "fivegrams": top_fivegrams,
                "total_unigrams": len(unigram_counts),
                "total_bigrams": len(bigram_counts),
                "total_trigrams": len(trigram_counts),
                "total_fourgrams": len(fourgram_counts),
                "total_fivegrams": len(fivegram_counts)
            }

            print(f"  - Unigrams: {len(top_unigrams)} (of {len(unigram_counts)} unique)")
            print(f"  - Bigrams: {len(top_bigrams)} (of {len(bigram_counts)} unique)")
            print(f"  - Trigrams: {len(top_trigrams)} (of {len(trigram_counts)} unique)")
            print(f"  - Fourgrams: {len(top_fourgrams)} (of {len(fourgram_counts)} unique)")
            print(f"  - Fivegrams: {len(top_fivegrams)} (of {len(fivegram_counts)} unique)")

        return features

    def save_features(self, features: Dict, output_path: Path) -> None:
        """
        Save features to JSON file

        Args:
            features: Feature dictionary
            output_path: Output file path
        """
        print(f"\nSaving features to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(f"Saved successfully!")

    def generate_and_save(
        self,
        auto_path: Path,
        manual_path: Path,
        overwrite_manual: bool = False
    ) -> Dict:
        """
        Generate features and save to both auto and manual files

        Args:
            auto_path: Path for auto_ngrams.json
            manual_path: Path for manual_ngrams.json
            overwrite_manual: If True, overwrite manual file even if it exists

        Returns:
            Generated features dictionary
        """
        # Load data
        self.load_training_data()

        # Generate features
        features = self.generate_ngram_features()

        # Always save to auto file
        self.save_features(features, auto_path)

        # Save to manual file (only if doesn't exist or overwrite is True)
        if overwrite_manual or not manual_path.exists():
            self.save_features(features, manual_path)
            print(f"\nManual file initialized at {manual_path}")
            print("You can now edit this file to customize n-gram features per tag!")
        else:
            print(f"\nManual file already exists at {manual_path}")
            print("Not overwriting. Use overwrite_manual=True to regenerate.")

        # Print summary
        print("\n" + "="*70)
        print("STS N-GRAM FEATURE EXTRACTION SUMMARY")
        print("="*70)
        print(f"Training samples: {len(self.train_df)}")
        print(f"Tags: {len(self.tags_sorted)}")
        print(f"Top-K per tag: {self.top_k}")
        print(f"Total n-grams: {len(self.tags_sorted) * self.top_k * 5}")
        print(f"Pattern dimension: {len(self.tags_sorted) * 5} (5 types × {len(self.tags_sorted)} tags)")
        print(f"Auto file: {auto_path}")
        print(f"Manual file: {manual_path}")
        print("="*70)

        return features


def load_ngram_features(features_path: Path) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Load n-gram features from JSON file for use in training/inference

    Args:
        features_path: Path to auto_ngrams.json or manual_ngrams.json

    Returns:
        Tuple of (tag_patterns, tags_sorted)
        - tag_patterns: {tag_name: {unigrams: set(), bigrams: set(), trigrams: set(), fourgrams: set(), fivegrams: set()}}
        - tags_sorted: Sorted list of tag names
    """
    print(f"Loading n-gram features from {features_path}")

    with open(features_path, 'r', encoding='utf-8') as f:
        features = json.load(f)

    tag_patterns = {}
    for tag_name, tag_data in features["tags"].items():
        tag_patterns[tag_name] = {
            "unigrams": set(item["ngram"] for item in tag_data.get("unigrams", [])),
            "bigrams": set(item["ngram"] for item in tag_data.get("bigrams", [])),
            "trigrams": set(item["ngram"] for item in tag_data["trigrams"]),
            "fourgrams": set(item["ngram"] for item in tag_data["fourgrams"]),
            "fivegrams": set(item["ngram"] for item in tag_data.get("fivegrams", []))
        }

    tags_sorted = sorted(tag_patterns.keys())

    print(f"Loaded features for {len(tags_sorted)} tags")
    print(f"Pattern dimension: {len(tags_sorted) * 5} (5 types × {len(tags_sorted)} tags)")

    # Show sample
    for tag in tags_sorted[:3]:
        print(f"  {tag}: {len(tag_patterns[tag]['unigrams'])} unigrams, "
              f"{len(tag_patterns[tag]['bigrams'])} bigrams, "
              f"{len(tag_patterns[tag]['trigrams'])} trigrams, "
              f"{len(tag_patterns[tag]['fourgrams'])} fourgrams, "
              f"{len(tag_patterns[tag]['fivegrams'])} fivegrams")

    return tag_patterns, tags_sorted
