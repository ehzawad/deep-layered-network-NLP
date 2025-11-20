#!/usr/bin/env python
"""
Verify pipelineNLP setup - check all required artifacts exist.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_python_version():
    """Check Python version is 3.12+"""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 12:
        return True, f"{major}.{minor}"
    return False, f"{major}.{minor} (need 3.12+)"


def check_device():
    """Check which device is available"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "mps"
        else:
            return True, "cpu"
    except ImportError:
        return False, "torch not installed"


def check_artifacts():
    """Check all required artifacts exist"""
    artifacts = [
        ("idea/models/semantic/faiss_index_global.index", "FAISS Index"),
        ("idea/models/semantic/sts_embeddings.npy", "STS Embeddings"),
        ("idea/models/semantic/question_mapping.csv", "Question Mapping"),
        ("idea/models/semantic/sts_metadata.json", "STS Metadata"),
        ("idea/models/tag_classifier/unified_tag_classifier.pth", "Classifier Model"),
        ("idea/models/tag_classifier/unified_tag_classifier_metadata.json", "Classifier Metadata"),
        ("idea/datasets/features/manual_ngrams.json", "N-gram Features"),
    ]

    results = []
    all_present = True

    for path_str, name in artifacts:
        path = Path(ROOT) / path_str
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            results.append((True, name, size_mb))
        else:
            results.append((False, name, 0))
            all_present = False

    return all_present, results


def main():
    print("pipelineNLP Setup Verification")
    print("=" * 80)

    # Check Python version
    py_ok, py_ver = check_python_version()
    print(f"Python version: {py_ver} {'✓' if py_ok else '✗'}")

    # Check device
    dev_ok, dev_name = check_device()
    print(f"Device: {dev_name} {'✓' if dev_ok else '✗'}")

    print()

    # Check artifacts
    all_ok, artifact_results = check_artifacts()
    print("Artifacts:")
    for present, name, size_mb in artifact_results:
        if present:
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: MISSING")

    print()

    if all_ok and py_ok and dev_ok:
        print("All required artifacts present! ✓")
        print()
        print("Next steps:")
        print("  python idea/examples/interactive_cli.py --top-k 2")
        return 0
    else:
        print("Setup incomplete! ✗")
        print()
        if not py_ok:
            print("  - Upgrade Python to 3.12+")
        if not dev_ok:
            print("  - Install PyTorch: pip install torch")
        if not all_ok:
            print("  - Run: python -c \"from idea.utils import artifacts; artifacts.ensure_all()\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
