# -*- coding: utf-8 -*-
"""
STS Training Script - Build FAISS Indices and Embeddings

Generates embeddings for all questions in the corpus and creates FAISS indices
for fast similarity search (global index only).

Uses shared model cache to avoid redundant model loading.
"""

import argparse
import os
import torch
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.config import E5PrefixConfig, DEFAULT_EMBEDDING_MODEL
from idea.utils.cache import compute_fingerprint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
_FAISS = None

if TYPE_CHECKING:  # pragma: no cover
    import faiss  # type: ignore


class STSTrainer:
    """
    STS Trainer - Build FAISS indices for semantic similarity search
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings: bool = True
    ):
        """
        Initialize STS trainer

        Args:
            embedding_model: Sentence transformer model name
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
        """
        self.embedding_model_name = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.embedding_model = None
        self.embedding_dim: int | None = None  # Will be set after encoding

    def load_data(self, train_file: Path, answer_file: Path):
        """
        Load training data and answers

        Args:
            train_file: Path to sts_train.csv
            answer_file: Path to tag_to_answer.json

        Returns:
            DataFrame with questions and tags
        """
        logger.info("Loading training data...")

        # Load questions and tags
        df = pd.read_csv(train_file)
        logger.info(f"  Loaded {len(df)} questions from {train_file.name}")
        logger.info(f"  Unique tags: {df['tag'].nunique()}")

        # Load answers
        with open(answer_file, 'r', encoding='utf-8') as f:
            self.tag_to_answer = json.load(f)
        logger.info(f"  Loaded {len(self.tag_to_answer)} answers")

        logger.info("\n  Sample tag distribution:")
        tag_counts = df['tag'].value_counts().head(10)
        for tag, count in tag_counts.items():
            logger.info(f"    {tag:40s}: {count:5d} questions")

        return df

    def generate_embeddings(self, questions):
        """
        Generate embeddings for all questions using shared model cache

        Args:
            questions: List of question strings

        Returns:
            numpy array of embeddings (n_questions, embedding_dim)
        """
        logger.info("\nGenerating embeddings...")
        logger.info(f"  Model: {self.embedding_model_name}")
        logger.info(f"  Questions: {len(questions)}")
        logger.info(f"  Normalize: {self.normalize_embeddings}")
        logger.info(f"  E5 prefixes enabled: {E5PrefixConfig.USE_PREFIXES}")
        if E5PrefixConfig.USE_PREFIXES:
            logger.info(f"  Passage prefix: '{E5PrefixConfig.STS_PASSAGE_PREFIX}'")
        
        # Apply E5-instruct passage prefix
        questions_to_encode = E5PrefixConfig.format_sts_passages_batch(questions)

        # Load model with device selection: CUDA → MPS → CPU
        from sentence_transformers import SentenceTransformer

        # Device selection: CUDA → MPS → CPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        logger.info(f"  Device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")

        logger.info(f"  Loading model on {device}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        logger.info(f"  ✓ Model loaded successfully on {device}")

        # Generate embeddings
        # Use larger batch size for GPU
        batch_size = 128 if device == 'cuda' else 32
        logger.info(f"  Encoding questions (this may take several minutes)...")
        logger.info(f"  Batch size: {batch_size}")

        embeddings = self.embedding_model.encode(
            questions_to_encode,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            batch_size=batch_size
        )

        # Persist the actual embedding dimension to avoid mismatches when changing
        # encoder backbones.
        self.embedding_dim = embeddings.shape[1]
        logger.info(f"  ✓ Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

        return embeddings

    def _lazy_import_faiss(self):
        global _FAISS
        if _FAISS is None:
            import faiss as _faiss_mod  # type: ignore
            _FAISS = _faiss_mod
        return _FAISS

    def build_faiss_indices(self, embeddings, indices_to_build='global'):
        """
        Build FAISS indices (global only)

        Args:
            embeddings: numpy array of embeddings (n_questions, embedding_dim)
            indices_to_build: retained for CLI compatibility (global only)

        Returns:
            Dict with single entry {'global': index}
        """
        logger.info("\nBuilding FAISS indices...")
        if self.embedding_dim is None:
            raise ValueError("embedding_dim not set; generate embeddings before building indices.")
        embeddings_f32 = embeddings.astype('float32')

        if indices_to_build not in ('all', 'global', None):
            logger.warning("Class-specific indices are no longer supported; building global index only.")

        logger.info("  Building global index...")
        embeddings_f32 = embeddings.astype('float32')

        start_time = time.time()
        faiss_mod = self._lazy_import_faiss()
        index_global = faiss_mod.IndexFlatIP(self.embedding_dim)
        index_global.add(embeddings_f32)
        build_time = time.time() - start_time

        logger.info(f"  ✓ Global index built: {index_global.ntotal:,} vectors in {build_time:.2f}s")
        return {'global': index_global}

    def save_artifacts(self, indices, embeddings, df, output_dir: Path, fingerprint: str | None = None):
        """
        Save all artifacts to disk

        Args:
            indices: Dict of FAISS indices
            embeddings: numpy array of embeddings
            df: DataFrame with questions + tags
            output_dir: Directory to save artifacts
        """
        logger.info("\nSaving artifacts...")
        faiss_mod = self._lazy_import_faiss()

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS indices
        logger.info("  Saving FAISS indices...")
        for name, index in indices.items():
            index_file = output_dir / f"faiss_index_{name}.index"
            faiss_mod.write_index(index, str(index_file))
            file_size_mb = index_file.stat().st_size / (1024 * 1024)
            logger.info(f"    ✓ {name:25s}: {index_file.name} ({file_size_mb:.2f} MB)")

        # 2. Save embeddings
        embeddings_file = output_dir / "sts_embeddings.npy"
        np.save(embeddings_file, embeddings)
        file_size_mb = embeddings_file.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Embeddings: {embeddings_file.name} ({file_size_mb:.2f} MB)")

        # 3. Save question-tag mapping
        mapping_file = output_dir / "question_mapping.csv"
        df[['question', 'tag']].to_csv(mapping_file, index=False)
        logger.info(f"  ✓ Question mapping: {mapping_file.name}")

        # 4. Save metadata
        metadata = {
            'model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.normalize_embeddings,
            'num_questions': len(df),
            'num_tags': int(df['tag'].nunique()),
            'num_classes': 0,
            'classes': [],
            'class_sizes': {},
            'created_at': datetime.now().isoformat(),
            'indices': {
                'global': indices['global'].ntotal,
            },
            'fingerprint': fingerprint,
            **E5PrefixConfig.get_metadata()
        }

        metadata_file = output_dir / "sts_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Metadata: {metadata_file.name}")

        logger.info(f"\n  All artifacts saved to: {output_dir}")

        return metadata


def train_sts(indices_to_build='all', force: bool = False):
    """
    Main training function

    Args:
        indices_to_build: 'all', 'global', or list of specific indices to build
    """
    print("=" * 80)
    print("STS TRAINING - BUILD FAISS INDICES")
    print("=" * 80)

    # Paths (IDEA-local)
    idea_dir = Path(__file__).parent.parent
    datasets_dir = idea_dir / "datasets"
    models_dir = idea_dir / "models" / "semantic"

    train_file = datasets_dir / "sts_train.csv"
    answer_file = datasets_dir / "tag_to_answer.json"

    # Initialize trainer
    trainer = STSTrainer(
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings=True
    )

    metadata_file = models_dir / "sts_metadata.json"
    cache_inputs = [train_file, answer_file]
    cache_extra = json.dumps({
        'embedding_model': trainer.embedding_model_name,
        'normalize_embeddings': trainer.normalize_embeddings,
        'indices_to_build': indices_to_build,
        'e5_prefix_config': E5PrefixConfig.get_cache_key()
    }, sort_keys=True)
    fingerprint = compute_fingerprint(cache_inputs, cache_extra)

    if not force and metadata_file.exists():
        try:
            with metadata_file.open('r', encoding='utf-8') as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = {}

        required_files = [
            models_dir / "faiss_index_global.index",
            models_dir / "sts_embeddings.npy",
            models_dir / "question_mapping.csv",
        ]
        if existing.get('fingerprint') == fingerprint and all(path.exists() for path in required_files):
            print("No changes detected; reusing cached FAISS artifacts. Use --force to rebuild.")
            return existing

    # Load data
    df = trainer.load_data(train_file, answer_file)

    # Generate embeddings (always need all embeddings even if building partial indices)
    embeddings = trainer.generate_embeddings(df['question'].tolist())

    # Build FAISS indices
    indices = trainer.build_faiss_indices(
        embeddings,
        indices_to_build=indices_to_build
    )

    # Save everything
    metadata = trainer.save_artifacts(indices, embeddings, df, models_dir, fingerprint=fingerprint)

    # Summary
    print("\n" + "=" * 80)
    print("✅ STS TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {models_dir}")
    print(f"\nStatistics:")
    print(f"  Questions:        {metadata['num_questions']:,}")
    print(f"  Tags:             {metadata['num_tags']}")
    print(f"  Embedding dim:    {metadata['embedding_dim']}")
    print(f"  Indices created:  {len(indices)}")
    print(f"\nIndices built:")
    for name, size in metadata['indices'].items():
        print(f"  {name:25s}: {size:,} vectors")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Build FAISS indices for STS component',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all indices (default)
  python idea/training/build_faiss_indices.py --all

  # Build only global index
  python idea/training/build_faiss_indices.py --global

  # Build specific indices (by name)
  python idea/training/build_faiss_indices.py --indices global accountProblems corrections
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all',
        action='store_true',
        help='Build the global index (default)'
    )
    group.add_argument(
        '--global',
        action='store_true',
        dest='global_only',
        help='Alias for --all (global index only)'
    )
    group.add_argument(
        '--indices',
        nargs='+',
        metavar='INDEX',
        help='Compatibility flag; only "global" is supported'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Ignore cache and rebuild artifacts'
    )

    args = parser.parse_args()

    # Determine which indices to build
    if args.global_only:
        indices_to_build = 'global'
    elif args.indices:
        indices_to_build = args.indices
    else:
        # Default: build all
        indices_to_build = 'all'

    train_sts(indices_to_build=indices_to_build, force=args.force)
