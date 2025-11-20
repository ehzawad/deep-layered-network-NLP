"""
STS N-gram Cleanup Script

Cleans noisy shared n-grams using dominance-based filtering. If no group
metadata is present, all tags are treated as belonging to a single group.

Algorithm:
1. Load auto_ngrams.json and overlap_analysis.json
2. For each tag group (group or the implicit "all_tags" group):
   - Evaluate overlaps and keep only dominant n-grams
3. Enforce a minimum number of n-grams per tag (default: 5)
4. Save to manual_ngrams.json with cleanup report
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class STSNgramCleaner:
    """Clean shared n-grams using within-group dominance-based filtering"""

    def __init__(
        self,
        auto_ngrams_path: Path,
        overlap_analysis_path: Path,
        dominance_ratio: float = 2.0,
        min_ngrams_per_tag: int = 5
    ):
        """
        Initialize cleaner

        Args:
            auto_ngrams_path: Path to auto_ngrams.json
            overlap_analysis_path: Path to overlap_analysis.json
            dominance_ratio: Minimum frequency ratio to keep shared n-gram (default: 2.0)
            min_ngrams_per_tag: Minimum n-grams to keep per tag (safety net)
        """
        self.auto_ngrams_path = auto_ngrams_path
        self.overlap_analysis_path = overlap_analysis_path
        self.dominance_ratio = dominance_ratio
        self.min_ngrams_per_tag = min_ngrams_per_tag

        self.features = None
        self.overlap_analysis = None
        self.tags_by_group = defaultdict(list)
        self.removed_ngrams = defaultdict(lambda: {"trigrams": [], "fourgrams": []})
        self.kept_shared = defaultdict(lambda: {"trigrams": [], "fourgrams": []})

    def load_data(self) -> None:
        """Load auto_ngrams.json and overlap_analysis.json"""
        print(f"Loading auto n-grams from: {self.auto_ngrams_path}")
        with open(self.auto_ngrams_path, 'r', encoding='utf-8') as f:
            self.features = json.load(f)

        print(f"Loading overlap analysis from: {self.overlap_analysis_path}")
        with open(self.overlap_analysis_path, 'r', encoding='utf-8') as f:
            self.overlap_analysis = json.load(f)

        # Group tags by group
        for tag, tag_data in self.features["tags"].items():
            group = tag_data.get("group", "all_tags")
            self.tags_by_group[group].append(tag)

        print(f"✅ Loaded features for {len(self.features['tags'])} tags")
        print(f"✅ Grouped into {len(self.tags_by_group)} groups")
        for group in sorted(self.tags_by_group.keys()):
            print(f"   {group}: {len(self.tags_by_group[group])} tags")

    def check_dominance(
        self,
        ngram: str,
        tag: str,
        group: str,
        ngram_type: str
    ) -> bool:
        """
        Check if tag dominates this n-gram within its group

        Args:
            ngram: The n-gram string
            tag: Tag name to check dominance for
            group: Group name
            ngram_type: "trigrams" or "fourgrams"

        Returns:
            True if tag has ≥dominance_ratio × max(other tags in group)
        """
        # Get overlap data for this group
        group_data = self.overlap_analysis["per_group"].get(group, {})
        overlap_data = group_data.get(f"{ngram_type[:-1]}_overlap", {})

        if ngram not in overlap_data:
            # Unique to this tag (no overlap)
            return True

        ngram_info = overlap_data[ngram]
        if ngram_info["overlap_count"] == 1:
            # Unique to this tag
            return True

        # Get frequencies for all tags in group with this n-gram
        frequencies = ngram_info["frequencies"]
        tag_freq = frequencies.get(tag, 0)

        if tag_freq == 0:
            return False

        # Get max frequency among OTHER tags in group
        other_freqs = [freq for t, freq in frequencies.items() if t != tag]
        if not other_freqs:
            return True

        max_other_freq = max(other_freqs)

        # Check dominance
        is_dominant = tag_freq >= (self.dominance_ratio * max_other_freq)

        return is_dominant

    def clean_group_ngrams(
        self,
        group: str,
        ngram_type: str
    ) -> Dict[str, Set[str]]:
        """
        Clean n-grams for all tags in a group

        Args:
            group: Group name
            ngram_type: "trigrams" or "fourgrams"

        Returns:
            Dictionary mapping tag → set of n-grams to KEEP
        """
        tags_in_group = self.tags_by_group[group]
        group_data = self.overlap_analysis["per_group"].get(group, {})
        overlap_data = group_data.get(f"{ngram_type[:-1]}_overlap", {})

        # Initialize: keep all n-grams initially
        tag_ngrams_to_keep = {}
        for tag in tags_in_group:
            tag_data = self.features["tags"][tag]
            all_ngrams = {item["ngram"] for item in tag_data[ngram_type]}
            tag_ngrams_to_keep[tag] = all_ngrams.copy()

        # Process shared n-grams (overlap_count > 1)
        shared_ngrams = [
            ngram for ngram, info in overlap_data.items()
            if info["overlap_count"] > 1
        ]

        for ngram in shared_ngrams:
            ngram_info = overlap_data[ngram]
            tags_with_ngram = ngram_info["tags"]

            # Check dominance for each tag
            for tag in tags_with_ngram:
                if tag not in tags_in_group:
                    continue  # Skip tags not in this group

                is_dominant = self.check_dominance(ngram, tag, group, ngram_type)

                if not is_dominant:
                    # Remove this n-gram from tag
                    if ngram in tag_ngrams_to_keep[tag]:
                        tag_ngrams_to_keep[tag].remove(ngram)
                        self.removed_ngrams[tag][ngram_type].append({
                            "ngram": ngram,
                            "group": group,
                            "frequencies": ngram_info["frequencies"],
                            "reason": "not_dominant"
                        })
                else:
                    # Keep this shared n-gram (it's dominant)
                    self.kept_shared[tag][ngram_type].append({
                        "ngram": ngram,
                        "group": group,
                        "frequencies": ngram_info["frequencies"]
                    })

        return tag_ngrams_to_keep

    def apply_safety_net(
        self,
        tag: str,
        ngrams_to_keep: Set[str],
        ngram_type: str
    ) -> Set[str]:
        """
        Ensure minimum n-grams per tag (safety net)

        Args:
            tag: Tag name
            ngrams_to_keep: Set of n-grams after cleanup
            ngram_type: "trigrams" or "fourgrams"

        Returns:
            Updated set with at least min_ngrams_per_tag entries
        """
        if len(ngrams_to_keep) >= self.min_ngrams_per_tag:
            return ngrams_to_keep

        # Need to restore some n-grams
        needed = self.min_ngrams_per_tag - len(ngrams_to_keep)
        tag_data = self.features["tags"][tag]
        original_ngrams = tag_data[ngram_type]

        # Restore top-ranked removed n-grams
        removed = [
            item for item in original_ngrams
            if item["ngram"] not in ngrams_to_keep
        ]
        removed_sorted = sorted(removed, key=lambda x: x["rank"])

        restored = set()
        for item in removed_sorted[:needed]:
            ngrams_to_keep.add(item["ngram"])
            restored.add(item["ngram"])

        if restored:
            print(f"  ⚠️  Safety net: Restored {len(restored)} {ngram_type} for {tag}")

        return ngrams_to_keep

    def clean_all_groups(self) -> Dict:
        """
        Clean n-grams for all groups

        Returns:
            Cleaned features dictionary
        """
        print("\n" + "="*70)
        print("CLEANING N-GRAMS (WITHIN-CLUSTER DOMINANCE-BASED)")
        print("="*70)
        print(f"Dominance ratio: {self.dominance_ratio}x")
        print(f"Min n-grams per tag: {self.min_ngrams_per_tag}")
        print("="*70)

        cleaned_features = {
            "metadata": self.features["metadata"].copy(),
            "tags": {}
        }

        # Add cleanup metadata
        cleaned_features["metadata"]["cleaned"] = True
        cleaned_features["metadata"]["dominance_ratio"] = self.dominance_ratio
        cleaned_features["metadata"]["min_ngrams_per_tag"] = self.min_ngrams_per_tag
        cleaned_features["metadata"]["cleanup_scope"] = "within_group"

        # Clean each group separately
        for group in sorted(self.tags_by_group.keys()):
            print(f"\n🔄 Cleaning group: {group}")
            tags_in_group = self.tags_by_group[group]
            print(f"   Tags: {len(tags_in_group)}")

            # Clean trigrams for this group
            trigrams_to_keep = self.clean_group_ngrams(group, "trigrams")

            # Clean fourgrams for this group
            fourgrams_to_keep = self.clean_group_ngrams(group, "fourgrams")

            # Apply safety net and build cleaned features
            for tag in tags_in_group:
                tag_data = self.features["tags"][tag]

                # Apply safety net
                trigrams_keep = self.apply_safety_net(
                    tag, trigrams_to_keep[tag], "trigrams"
                )
                fourgrams_keep = self.apply_safety_net(
                    tag, fourgrams_to_keep[tag], "fourgrams"
                )

                # Build cleaned n-gram lists (preserve rank order)
                cleaned_trigrams = [
                    item for item in tag_data["trigrams"]
                    if item["ngram"] in trigrams_keep
                ]
                cleaned_fourgrams = [
                    item for item in tag_data["fourgrams"]
                    if item["ngram"] in fourgrams_keep
                ]

                cleaned_features["tags"][tag] = {
                    "trigrams": cleaned_trigrams,
                    "fourgrams": cleaned_fourgrams,
                    "group": tag_data["group"],
                    "total_trigrams": tag_data["total_trigrams"],
                    "total_fourgrams": tag_data["total_fourgrams"]
                }

                # Log removal stats
                removed_tg = len(tag_data["trigrams"]) - len(cleaned_trigrams)
                removed_fg = len(tag_data["fourgrams"]) - len(cleaned_fourgrams)

                if removed_tg > 0 or removed_fg > 0:
                    print(f"   {tag}: removed {removed_tg} trigrams, {removed_fg} fourgrams")

        print("\n" + "="*70)
        print("✅ Cleanup complete!")
        print("="*70)

        return cleaned_features

    def generate_cleanup_report(self, cleaned_features: Dict) -> Dict:
        """
        Generate detailed cleanup report

        Args:
            cleaned_features: Cleaned features dictionary

        Returns:
            Cleanup report dictionary
        """
        print("\n📊 Generating cleanup report...")

        # Count totals
        original_trigrams = sum(
            len(tag_data["trigrams"])
            for tag_data in self.features["tags"].values()
        )
        original_fourgrams = sum(
            len(tag_data["fourgrams"])
            for tag_data in self.features["tags"].values()
        )

        cleaned_trigrams = sum(
            len(tag_data["trigrams"])
            for tag_data in cleaned_features["tags"].values()
        )
        cleaned_fourgrams = sum(
            len(tag_data["fourgrams"])
            for tag_data in cleaned_features["tags"].values()
        )

        removed_trigrams = original_trigrams - cleaned_trigrams
        removed_fourgrams = original_fourgrams - cleaned_fourgrams

        # Per-group statistics
        per_group_stats = {}
        for group in sorted(self.tags_by_group.keys()):
            tags_in_group = self.tags_by_group[group]

            group_orig_tg = sum(
                len(self.features["tags"][tag]["trigrams"])
                for tag in tags_in_group
            )
            group_orig_fg = sum(
                len(self.features["tags"][tag]["fourgrams"])
                for tag in tags_in_group
            )

            group_clean_tg = sum(
                len(cleaned_features["tags"][tag]["trigrams"])
                for tag in tags_in_group
            )
            group_clean_fg = sum(
                len(cleaned_features["tags"][tag]["fourgrams"])
                for tag in tags_in_group
            )

            per_group_stats[group] = {
                "tags": len(tags_in_group),
                "trigrams": {
                    "original": group_orig_tg,
                    "cleaned": group_clean_tg,
                    "removed": group_orig_tg - group_clean_tg,
                    "removed_pct": (group_orig_tg - group_clean_tg) / group_orig_tg * 100
                    if group_orig_tg > 0 else 0
                },
                "fourgrams": {
                    "original": group_orig_fg,
                    "cleaned": group_clean_fg,
                    "removed": group_orig_fg - group_clean_fg,
                    "removed_pct": (group_orig_fg - group_clean_fg) / group_orig_fg * 100
                    if group_orig_fg > 0 else 0
                }
            }

        # Per-tag details
        per_tag_details = {}
        for tag in sorted(self.features["tags"].keys()):
            per_tag_details[tag] = {
                "group": self.features["tags"][tag]["group"],
                "trigrams": {
                    "original": len(self.features["tags"][tag]["trigrams"]),
                    "cleaned": len(cleaned_features["tags"][tag]["trigrams"]),
                    "removed": len(self.removed_ngrams[tag]["trigrams"]),
                    "kept_shared": len(self.kept_shared[tag]["trigrams"])
                },
                "fourgrams": {
                    "original": len(self.features["tags"][tag]["fourgrams"]),
                    "cleaned": len(cleaned_features["tags"][tag]["fourgrams"]),
                    "removed": len(self.removed_ngrams[tag]["fourgrams"]),
                    "kept_shared": len(self.kept_shared[tag]["fourgrams"])
                },
                "removed_ngrams": self.removed_ngrams[tag],
                "kept_shared_ngrams": self.kept_shared[tag]
            }

        report = {
            "metadata": {
                "dominance_ratio": self.dominance_ratio,
                "min_ngrams_per_tag": self.min_ngrams_per_tag,
                "cleanup_scope": "within_group"
            },
            "summary": {
                "total_tags": len(self.features["tags"]),
                "total_groups": len(self.tags_by_group),
                "trigrams": {
                    "original": original_trigrams,
                    "cleaned": cleaned_trigrams,
                    "removed": removed_trigrams,
                    "removed_pct": removed_trigrams / original_trigrams * 100
                    if original_trigrams > 0 else 0
                },
                "fourgrams": {
                    "original": original_fourgrams,
                    "cleaned": cleaned_fourgrams,
                    "removed": removed_fourgrams,
                    "removed_pct": removed_fourgrams / original_fourgrams * 100
                    if original_fourgrams > 0 else 0
                }
            },
            "per_group": per_group_stats,
            "per_tag": per_tag_details
        }

        return report

    def print_summary(self, report: Dict) -> None:
        """Print human-readable summary"""
        print("\n" + "="*70)
        print("CLEANUP REPORT SUMMARY")
        print("="*70)

        summary = report["summary"]
        print(f"\n📈 GLOBAL STATISTICS:")
        print(f"   Total tags: {summary['total_tags']}")
        print(f"   Total groups: {summary['total_groups']}")
        print(f"   Dominance ratio: {report['metadata']['dominance_ratio']}x")

        print(f"\n   TRIGRAMS:")
        print(f"      Original: {summary['trigrams']['original']}")
        print(f"      Cleaned:  {summary['trigrams']['cleaned']}")
        print(f"      Removed:  {summary['trigrams']['removed']} "
              f"({summary['trigrams']['removed_pct']:.1f}%)")

        print(f"\n   FOURGRAMS:")
        print(f"      Original: {summary['fourgrams']['original']}")
        print(f"      Cleaned:  {summary['fourgrams']['cleaned']}")
        print(f"      Removed:  {summary['fourgrams']['removed']} "
              f"({summary['fourgrams']['removed_pct']:.1f}%)")

        print(f"\n📊 PER-CLUSTER BREAKDOWN:")
        for group in sorted(report["per_group"].keys()):
            group_stats = report["per_group"][group]
            print(f"\n   {group} ({group_stats['tags']} tags):")
            print(f"      Trigrams:  {group_stats['trigrams']['removed']}/{group_stats['trigrams']['original']} removed "
                  f"({group_stats['trigrams']['removed_pct']:.1f}%)")
            print(f"      Fourgrams: {group_stats['fourgrams']['removed']}/{group_stats['fourgrams']['original']} removed "
                  f"({group_stats['fourgrams']['removed_pct']:.1f}%)")

        print("\n" + "="*70)

    def save_cleaned_features(self, cleaned_features: Dict, output_path: Path) -> None:
        """Save cleaned features to manual_ngrams.json"""
        print(f"\n💾 Saving cleaned features to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_features, f, ensure_ascii=False, indent=2)
        print("✅ Saved successfully!")

    def save_cleanup_report(self, report: Dict, output_path: Path) -> None:
        """Save cleanup report to JSON"""
        print(f"💾 Saving cleanup report to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("✅ Saved successfully!")

    def clean_and_save(
        self,
        manual_ngrams_path: Path,
        cleanup_report_path: Path
    ) -> Tuple[Dict, Dict]:
        """
        Complete cleanup workflow

        Args:
            manual_ngrams_path: Path to save cleaned manual_ngrams.json
            cleanup_report_path: Path to save cleanup_report.json

        Returns:
            Tuple of (cleaned_features, cleanup_report)
        """
        # Load data
        self.load_data()

        # Clean
        cleaned_features = self.clean_all_groups()

        # Generate report
        report = self.generate_cleanup_report(cleaned_features)

        # Print summary
        self.print_summary(report)

        # Save
        self.save_cleaned_features(cleaned_features, manual_ngrams_path)
        self.save_cleanup_report(report, cleanup_report_path)

        return cleaned_features, report
