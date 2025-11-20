"""
STS N-gram Feature Overlap Analyzer

Analyzes cross-tag n-gram overlaps within logical groups (groups). When
group metadata is absent, all tags are analyzed together under a single group.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

class STSFeatureAnalyzer:
    """Analyze n-gram feature overlaps within groups"""

    def __init__(self, features_path: Path):
        """
        Initialize feature analyzer

        Args:
            features_path: Path to ngrams JSON file (auto or manual)
        """
        self.features_path = features_path
        self.features = None
        self.tags_by_group = defaultdict(list)

    def load_features(self) -> None:
        """Load n-gram features from JSON file"""
        print(f"Loading features from {self.features_path}")
        with open(self.features_path, 'r', encoding='utf-8') as f:
            self.features = json.load(f)

        # Group tags by group
        for tag, tag_data in self.features["tags"].items():
            group = tag_data.get("group", "all_tags")
            self.tags_by_group[group].append(tag)

        print(f"Loaded features for {len(self.features['tags'])} tags")
        print(f"Grouped into {len(self.tags_by_group)} groups")
        for group, tags in sorted(self.tags_by_group.items()):
            print(f"  {group}: {len(tags)} tags")

    def analyze_group_overlap(self, group: str, ngram_type: str) -> Dict:
        """
        Analyze overlap for specific n-gram type within a group

        Args:
            group: Group name
            ngram_type: "trigrams", "fourgrams", or "fivegrams"

        Returns:
            Dictionary with overlap analysis within this group only
        """
        tags_in_group = self.tags_by_group[group]

        # Build inverted index: ngram → {tag: frequency} (within group only)
        ngram_to_tags = defaultdict(dict)

        for tag in tags_in_group:
            tag_data = self.features["tags"][tag]
            for item in tag_data[ngram_type]:
                ngram = item["ngram"]
                frequency = item["frequency"]
                ngram_to_tags[ngram][tag] = frequency

        # Analyze overlaps
        overlap_analysis = {}
        for ngram, tag_freqs in ngram_to_tags.items():
            overlap_count = len(tag_freqs)
            overlap_analysis[ngram] = {
                "tags": sorted(tag_freqs.keys()),
                "frequencies": tag_freqs,
                "overlap_count": overlap_count,
                "overlap_type": "unique" if overlap_count == 1 else "shared",
                "group": group
            }

        return overlap_analysis

    def generate_overlap_analysis(self) -> Dict:
        """
        Generate complete overlap analysis per group

        Returns:
            Dictionary with per-group analysis
        """
        print("\n" + "="*70)
        print("ANALYZING N-GRAM OVERLAPS (WITHIN-CLUSTER ONLY)")
        print("="*70)

        analysis = {
            "metadata": {
                "source_file": str(self.features_path),
                "num_tags": len(self.features["tags"]),
                "num_groups": len(self.tags_by_group),
                "analysis_scope": "within_group"
            },
            "per_group": {}
        }

        # Analyze each group separately
        for group in sorted(self.tags_by_group.keys()):
            print(f"\nAnalyzing group: {group}")

            tags_in_group = self.tags_by_group[group]
            print(f"  Tags in group: {len(tags_in_group)}")

            # Analyze trigrams
            trigram_overlap = self.analyze_group_overlap(group, "trigrams")
            print(f"  Unique trigrams in group: {len(trigram_overlap)}")

            # Analyze fourgrams
            fourgram_overlap = self.analyze_group_overlap(group, "fourgrams")
            print(f"  Unique fourgrams in group: {len(fourgram_overlap)}")

            # Analyze fivegrams
            fivegram_overlap = self.analyze_group_overlap(group, "fivegrams")
            print(f"  Unique fivegrams in group: {len(fivegram_overlap)}")

            # Compute statistics
            stats = self._compute_group_statistics(
                group,
                tags_in_group,
                trigram_overlap,
                fourgram_overlap,
                fivegram_overlap
            )

            analysis["per_group"][group] = {
                "tags": tags_in_group,
                "trigram_overlap": trigram_overlap,
                "fourgram_overlap": fourgram_overlap,
                "fivegram_overlap": fivegram_overlap,
                "statistics": stats
            }

        # Global summary
        analysis["global_summary"] = self._compute_global_summary(analysis)

        return analysis

    def _compute_group_statistics(
        self,
        group: str,
        tags_in_group: List[str],
        trigram_overlap: Dict,
        fourgram_overlap: Dict,
        fivegram_overlap: Dict
    ) -> Dict:
        """Compute statistics for a group"""

        stats = {
            "num_tags": len(tags_in_group),
            "trigrams": {
                "total_unique": len(trigram_overlap),
                "tag_specific": sum(
                    1 for v in trigram_overlap.values()
                    if v["overlap_count"] == 1
                ),
                "shared": sum(
                    1 for v in trigram_overlap.values()
                    if v["overlap_count"] > 1
                ),
                "max_overlap": max(
                    (v["overlap_count"] for v in trigram_overlap.values()),
                    default=0
                )
            },
            "fourgrams": {
                "total_unique": len(fourgram_overlap),
                "tag_specific": sum(
                    1 for v in fourgram_overlap.values()
                    if v["overlap_count"] == 1
                ),
                "shared": sum(
                    1 for v in fourgram_overlap.values()
                    if v["overlap_count"] > 1
                ),
                "max_overlap": max(
                    (v["overlap_count"] for v in fourgram_overlap.values()),
                    default=0
                )
            },
            "fivegrams": {
                "total_unique": len(fivegram_overlap),
                "tag_specific": sum(
                    1 for v in fivegram_overlap.values()
                    if v["overlap_count"] == 1
                ),
                "shared": sum(
                    1 for v in fivegram_overlap.values()
                    if v["overlap_count"] > 1
                ),
                "max_overlap": max(
                    (v["overlap_count"] for v in fivegram_overlap.values()),
                    default=0
                )
            }
        }

        # Per-tag statistics
        stats["per_tag"] = {}
        for tag in tags_in_group:
            tag_data = self.features["tags"][tag]

            # Count unique vs shared for this tag
            trigrams_unique = sum(
                1 for item in tag_data["trigrams"]
                if trigram_overlap[item["ngram"]]["overlap_count"] == 1
            )
            fourgrams_unique = sum(
                1 for item in tag_data["fourgrams"]
                if fourgram_overlap[item["ngram"]]["overlap_count"] == 1
            )
            fivegrams_unique = sum(
                1 for item in tag_data.get("fivegrams", [])
                if item["ngram"] in fivegram_overlap and fivegram_overlap[item["ngram"]]["overlap_count"] == 1
            )

            stats["per_tag"][tag] = {
                "trigrams_total": len(tag_data["trigrams"]),
                "trigrams_unique": trigrams_unique,
                "trigrams_shared": len(tag_data["trigrams"]) - trigrams_unique,
                "fourgrams_total": len(tag_data["fourgrams"]),
                "fourgrams_unique": fourgrams_unique,
                "fourgrams_shared": len(tag_data["fourgrams"]) - fourgrams_unique,
                "fivegrams_total": len(tag_data.get("fivegrams", [])),
                "fivegrams_unique": fivegrams_unique,
                "fivegrams_shared": len(tag_data.get("fivegrams", [])) - fivegrams_unique
            }

        # Most confused tag pairs (within group)
        tag_pair_overlaps = defaultdict(int)
        for ngram_data in list(trigram_overlap.values()) + list(fourgram_overlap.values()) + list(fivegram_overlap.values()):
            if ngram_data["overlap_count"] > 1:
                tags = ngram_data["tags"]
                for i in range(len(tags)):
                    for j in range(i + 1, len(tags)):
                        pair = tuple(sorted([tags[i], tags[j]]))
                        tag_pair_overlaps[pair] += 1

        top_overlaps = sorted(
            tag_pair_overlaps.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        stats["top_tag_overlaps"] = [
            {"tags": list(pair), "shared_ngrams": count}
            for pair, count in top_overlaps
        ]

        return stats

    def _compute_global_summary(self, analysis: Dict) -> Dict:
        """Compute global summary across all groups"""
        total_tags = 0
        total_trigrams = 0
        total_fourgrams = 0
        total_fivegrams = 0
        total_trigrams_shared = 0
        total_fourgrams_shared = 0
        total_fivegrams_shared = 0

        for group_data in analysis["per_group"].values():
            stats = group_data["statistics"]
            total_tags += stats["num_tags"]
            total_trigrams += stats["trigrams"]["total_unique"]
            total_fourgrams += stats["fourgrams"]["total_unique"]
            total_fivegrams += stats["fivegrams"]["total_unique"]
            total_trigrams_shared += stats["trigrams"]["shared"]
            total_fourgrams_shared += stats["fourgrams"]["shared"]
            total_fivegrams_shared += stats["fivegrams"]["shared"]

        return {
            "total_tags": total_tags,
            "total_groups": len(analysis["per_group"]),
            "trigrams": {
                "total_unique": total_trigrams,
                "shared": total_trigrams_shared,
                "shared_pct": (total_trigrams_shared / total_trigrams * 100) if total_trigrams > 0 else 0
            },
            "fourgrams": {
                "total_unique": total_fourgrams,
                "shared": total_fourgrams_shared,
                "shared_pct": (total_fourgrams_shared / total_fourgrams * 100) if total_fourgrams > 0 else 0
            },
            "fivegrams": {
                "total_unique": total_fivegrams,
                "shared": total_fivegrams_shared,
                "shared_pct": (total_fivegrams_shared / total_fivegrams * 100) if total_fivegrams > 0 else 0
            }
        }

    def save_analysis(self, analysis: Dict, output_path: Path) -> None:
        """Save overlap analysis to JSON file"""
        print(f"\nSaving overlap analysis to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print("Saved successfully!")

    def print_summary(self, analysis: Dict) -> None:
        """Print human-readable summary"""
        print("\n" + "="*70)
        print("N-GRAM OVERLAP ANALYSIS SUMMARY (WITHIN-CLUSTER)")
        print("="*70)

        summary = analysis["global_summary"]

        print(f"\nGLOBAL STATISTICS:")
        print(f"  Total tags: {summary['total_tags']}")
        print(f"  Total groups: {summary['total_groups']}")

        print(f"\n  TRIGRAMS (across all groups):")
        print(f"    Total unique: {summary['trigrams']['total_unique']}")
        print(f"    Shared within groups: {summary['trigrams']['shared']} "
              f"({summary['trigrams']['shared_pct']:.1f}%)")

        print(f"\n  FOURGRAMS (across all groups):")
        print(f"    Total unique: {summary['fourgrams']['total_unique']}")
        print(f"    Shared within groups: {summary['fourgrams']['shared']} "
              f"({summary['fourgrams']['shared_pct']:.1f}%)")

        print(f"\n  FIVEGRAMS (across all groups):")
        print(f"    Total unique: {summary['fivegrams']['total_unique']}")
        print(f"    Shared within groups: {summary['fivegrams']['shared']} "
              f"({summary['fivegrams']['shared_pct']:.1f}%)")

        print(f"\nPER-CLUSTER BREAKDOWN:")
        for group in sorted(analysis["per_group"].keys()):
            group_data = analysis["per_group"][group]
            stats = group_data["statistics"]

            print(f"\n  {group}:")
            print(f"    Tags: {stats['num_tags']}")
            print(f"    Trigrams: {stats['trigrams']['total_unique']} unique "
                  f"({stats['trigrams']['tag_specific']} tag-specific, "
                  f"{stats['trigrams']['shared']} shared)")
            print(f"    Fourgrams: {stats['fourgrams']['total_unique']} unique "
                  f"({stats['fourgrams']['tag_specific']} tag-specific, "
                  f"{stats['fourgrams']['shared']} shared)")
            print(f"    Fivegrams: {stats['fivegrams']['total_unique']} unique "
                  f"({stats['fivegrams']['tag_specific']} tag-specific, "
                  f"{stats['fivegrams']['shared']} shared)")

            if stats["top_tag_overlaps"]:
                print(f"    Top confused pairs:")
                for item in stats["top_tag_overlaps"][:3]:
                    print(f"      {item['tags'][0]} <-> {item['tags'][1]}: "
                          f"{item['shared_ngrams']} shared")

        print("\n" + "="*70)

    def analyze_and_save(self, output_path: Path) -> Dict:
        """Load features, analyze, save, and print summary"""
        self.load_features()
        analysis = self.generate_overlap_analysis()
        self.save_analysis(analysis, output_path)
        self.print_summary(analysis)
        return analysis
