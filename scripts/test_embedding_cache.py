#!/usr/bin/env python
"""
Quick test to verify embedding caching optimization.

Usage:
    python scripts/test_embedding_cache.py
"""

import sys
import time
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea import IdeaPipeline

def test_embedding_cache():
    print("=" * 80)
    print("TESTING EMBEDDING CACHE OPTIMIZATION")
    print("=" * 80)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = IdeaPipeline()
    pipeline.initialize()

    # Test query
    test_question = "আমার জাতীয় পরিচয়পত্রের ঠিকানা পরিবর্তন করতে চাই"

    print(f"\n2. Running query: {test_question[:50]}...")
    start = time.time()
    result = pipeline.run(test_question, fusion_top_k=5)
    elapsed = time.time() - start

    print(f"\n3. Results:")
    print(f"   Primary answer tag: {result['primary_tag']}")
    print(f"   Total latency: {result['telemetry']['latency_ms']:.2f}ms")
    print(f"\n4. Checking embedding reuse:")

    # Check if both matchers used precomputed embedding
    sts_metadata = result['signals']['sts']['metadata']
    used_cache = sts_metadata.get('used_precomputed_embedding', False)

    if used_cache:
        print("   ✅ SUCCESS! FAISS used pre-computed embedding (no duplicate encoding)")
    else:
        print("   ❌ FAISS did NOT use cached embedding")

    print(f"\n5. Top 5 candidates:")
    for i, cand in enumerate(result['candidates'][:5], 1):
        print(f"   {i}. {cand['tag'][:50]:50s} (score: {cand['final_score']:.4f}, source: {cand['source']})")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_embedding_cache()
