# pipelineNLP

Experimental dual-signal question answering system for National ID card support queries using FAISS semantic search and 203-tag neural classifier with weighted fusion.

---

## Quick Start

```bash
# Fresh installation (takes 30-60 minutes on first run)
git clone https://github.com/ehzawad/pipelineNLP.git && cd pipelineNLP
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "from idea.utils import artifacts; artifacts.ensure_all()"

# Verify setup
python scripts/verify_setup.py

# Test interactively
python idea/examples/interactive_cli.py --top-k 1

# Run evaluation
python idea/evals/eval_pipeline.py --data idea/datasets/sts_eval.csv --top-k 1
```

**Requirements:** Python 3.12+, 5GB disk space, internet connection for first-time model download (~1.2GB)

---

## Architecture

### Dual-Signal Fusion Pipeline

```
User Question
      │
      ├──────────────────────┬───────────────────────┐
      │                      │                       │
      ▼                      ▼                       ▼
[SemanticSearchEngine] [TagClassifier]     [Config: E5PrefixConfig]
      │                      │                       │
  (FAISS Matcher)    (Classifier Matcher)    (Centralized prefix logic)
      │                      │
  - Encodes query       - Encodes query
    with "query: "        with "query: "
  - Searches FAISS      - Runs through
    index built from      203-tag model
    "passage: "          - Returns top-K
    prefixed corpus       tag predictions
  - Returns top-K
    similar questions
      │                      │
      └──────────┬───────────┘
                 ▼
         [DualSignalRanker]
                 │
         - Normalizes STS scores using
           per-query min-max normalization
         - Fuses signals with weights:
           * STS: 0.49
           * Classifier: 0.51
         - Drops candidates below min_score
         - Returns ranked answers
                 │
                 ▼
           Final Answer
```

### How It Works

1. **SemanticSearchEngine** - Encodes question with E5-instruct prefixes, searches global FAISS index, returns top-N similar questions
2. **TagClassifier** - Uses UnifiedTagClassifier (embedding + n-gram features), predicts across 203 tags with confidence scores
3. **DualSignalRanker** - Normalizes both signals per-query (min-max), fuses with configurable weights, filters by threshold
4. **IdeaPipeline** - Orchestrates execution, handles fallbacks, returns structured response with telemetry

**Output structure:**
```python
{
    "question": "...",
    "answer": "primary answer text",
    "primary_tag": "TAG_NAME",
    "candidates": [...],          # fused + filtered results
    "dropped_candidates": [...],  # filtered out by min_score threshold
    "signals": {
        "sts": {"results": [...], "metadata": {...}},
        "classifier": {"results": [...], "metadata": {...}}
    },
    "telemetry": {...}
}
```

---

## Project Structure

```
pipelineNLP/
├── architecturaldecision/          # Design docs & ADRs
│   └── PipelineInception.pdf       # System inception notes
├── idea/                           # Main package (historical name)
│   ├── config.py                   # Centralized configuration (E5 prefixes, weights, top-k)
│   ├── pipeline.py                 # Main orchestrator (IdeaPipeline)
│   ├── ranker.py                   # Signal fusion logic (DualSignalRanker)
│   ├── semantic_similarity.py      # SemanticSearchEngine wrapper
│   ├── tag_classifier.py           # TagClassifier wrapper
│   ├── inference/                  # Model inference code
│   │   ├── faiss_matcher.py        # FAISS semantic search
│   │   ├── tag_classifier_matcher.py # Tag prediction
│   │   └── unified_tag_classifier.py # Hybrid neural classifier
│   ├── training/                   # Training scripts
│   │   ├── build_faiss_indices.py  # Build FAISS index
│   │   ├── train_tag_classifier.py # Train 203-tag classifier
│   │   └── unified_tag_classifier_trainer.py # Trainer implementation
│   ├── featurizer/                 # N-gram feature generation
│   │   ├── generate_features.py    # Main feature pipeline
│   │   ├── ngram_extractor.py      # Extract n-grams per tag
│   │   ├── feature_analyzer.py     # Overlap analysis
│   │   └── clean_ngrams.py         # Dominance-based filtering
│   ├── evals/                      # Evaluation tools
│   │   ├── eval_pipeline.py        # Batch evaluation script
│   │   └── simplify_csv.py         # CSV simplification
│   ├── datasets/                   # Training/eval data
│   │   ├── sts_train.csv           # 36,358 training questions (question,tag)
│   │   ├── sts_eval.csv            # 7,794 eval questions (question,tag)
│   │   ├── tag_to_answer.json      # 203 tag→answer mappings consumed at runtime
│   │   ├── irrelevant.csv          # Optional chitchat guardrail set for the `irrelevant` tag
│   │   └── features/               # Generated n-gram features (auto + manual)
│   │       ├── manual_ngrams.json  # Primary pattern file used by classifier
│   │       ├── auto_ngrams.json    # Auto-generated patterns (reference/experiments)
│   │       ├── overlap_analysis.json # Tag overlap diagnostics
│   │       ├── cleanup_report.json # Dominance filtering report
│   │       └── manual_ngrams_top40_backup.json # Snapshot before manual pruning
│   ├── models/                     # Trained artifacts
│   │   ├── semantic/               # FAISS index + embeddings
│   │   └── tag_classifier/         # PyTorch weights + metadata
│   ├── utils/                      # Shared utilities
│   │   ├── artifacts.py            # Auto-build missing artifacts
│   │   ├── model_cache.py          # Shared SentenceTransformer cache
│   │   └── data_utils/             # Data cleaning tools
│   └── examples/
│       └── interactive_cli.py      # Interactive testing
└── scripts/
    └── verify_setup.py             # Installation verification
```

---

## Installation & Setup

### Prerequisites

- **Python 3.12+** (check: `python --version`)
- **5GB free disk space**
- **Internet connection** (first-time model download)
- **Optional:** NVIDIA GPU (CUDA) or Apple Silicon (M1/M2/M3) for faster inference

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ehzawad/pipelineNLP.git
cd pipelineNLP

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: GPU acceleration
# Apple Silicon:
export STS_EMBEDDING_DEVICE=mps

# NVIDIA GPU:
pip uninstall faiss-cpu
pip install faiss-gpu
export STS_EMBEDDING_DEVICE=cuda
```

### Build Artifacts (First-Time Setup)

**Automatic build (recommended):**
```bash
python -c "from idea.utils import artifacts; artifacts.ensure_all()"
```

Takes 30-60 minutes on first run:
- Downloads `intfloat/multilingual-e5-large-instruct` (~1.2GB)
- Generates 606-dim n-gram features from 36,358 questions
- Trains 203-tag classifier (50 epochs with early stopping)
- Builds FAISS index with 36,358 embeddings

**Manual build (advanced):**
```bash
# 1. Generate n-gram features
python idea/featurizer/generate_features.py \
  --top-k 20 --auto-clean --dominance-ratio 2.0 --force-overwrite

# 2. Build FAISS semantic index
python idea/training/build_faiss_indices.py --global --force

# 3. Train 203-tag classifier
python idea/training/train_tag_classifier.py \
  --models idea/models/tag_classifier \
  --embedding-model intfloat/multilingual-e5-large-instruct \
  --epochs 50 --batch-size 64 --lr 3e-4 --patience 12 \
  --ngram-mode manual --force
```

### Verify Installation

```bash
python scripts/verify_setup.py
```

Expected output:
```
pipelineNLP Setup Verification
================================================================================
Python version: 3.12.x ✓
Device: mps ✓

Artifacts:
  ✓ FAISS Index: 136.0 MB
  ✓ STS Embeddings: 138.7 MB
  ✓ Classifier Model: 3.3 MB
  ✓ N-gram Features: 0.9 MB

All required artifacts present! ✓
```

---

## Usage

### Interactive CLI

```bash
python idea/examples/interactive_cli.py --top-k 1
```

Example session:
```
Interactive pipelineNLP (top-2; type 'quit' to exit)
Question> আমার স্মার্ট কার্ড কবে আসবে?
Top tag: smart_card_status
Candidates:
  1. smart_card_status (score=0.842, source=sts+classifier)
  2. smart_card_delivery (score=0.531, source=sts+classifier)
```

### Programmatic Usage

```python
from idea import IdeaPipeline, IdeaConfig

# Initialize pipeline
pipe = IdeaPipeline(IdeaConfig())
pipe.initialize()

# Run query
result = pipe.run("How do I change my MFA device?", fusion_top_k=3)

# Access results
print(result["answer"])          # Primary answer text
print(result["primary_tag"])     # Predicted tag
print(result["candidates"])      # Ranked candidates with scores

# Debug with telemetry
print(result["telemetry"]["ranker"])      # Weights, normalization stats
print(result["dropped_candidates"])        # Filtered candidates
```

### Batch Evaluation

```bash
python idea/evals/eval_pipeline.py \
  --data idea/datasets/sts_eval.csv \
  --top-k 1 \
  --output-csv idea/evals/results/eval_latest.csv
```

Output includes:
- Top-1/Top-K accuracy metrics
- Per-query predictions with sources (STS vs classifier vs fused)
- Cosine similarity scores from FAISS
- All fused candidates before/after threshold filtering

**Simplify results to core columns:**
```bash
python idea/evals/simplify_csv.py \
  idea/evals/results/eval_latest.csv \
  idea/evals/results/eval_latest_simple.csv
```

---

## Configuration

All settings in `idea/config.py`:

### E5 Prefix Configuration

**Centralized in `E5PrefixConfig` class** (single source of truth):

```python
USE_PREFIXES = True          # Enable/disable E5-instruct prefixes
STS_QUERY_PREFIX = "query: "     # FAISS query prefix
STS_PASSAGE_PREFIX = "passage: " # FAISS corpus prefix (at index build time)
CLASSIFIER_PREFIX = "query: "    # Classifier prefix (train + inference)
```

**Important:** Changing prefixes requires rebuilding ALL artifacts with `--force` flag.

### Top-K Configuration

Three independent stages:

1. **`SemanticSearchConfig.top_k`** (default: 5) - FAISS recall depth
2. **`TagClassifierConfig.top_k`** (default: 25) - Number of classifier predictions
3. **`IdeaConfig.fusion_top_k`** (default: 10) - Final output limit (runtime override available)

### Ranker Weights

```python
# In RankerConfig
weights: Dict[str, float] = {
    "sts": 0.49,        # FAISS semantic search weight
    "classifier": 0.51  # Tag classifier weight
}
min_score: float = 0.05  # Threshold for filtering candidates
# Hard floors (raw, pre-normalization)
min_sts_similarity: float = 0.2
min_classifier_confidence: float = 0.2
# Allow abstain when both signals are weak
abstain_on_low_signals: bool = True
abstain_answer: str = "I can't answer that with enough confidence right now."
```

No rebuild needed for weight changes - applied at inference time.

### Device Configuration

Set environment variable:
```bash
export STS_EMBEDDING_DEVICE=auto   # auto-detect (default)
export STS_EMBEDDING_DEVICE=cuda   # NVIDIA GPU
export STS_EMBEDDING_DEVICE=mps    # Apple Silicon
export STS_EMBEDDING_DEVICE=cpu    # CPU only
```

---

## Training & Artifacts

### N-Gram Feature Generation

```bash
python idea/featurizer/generate_features.py \
  --top-k 20 \
  --auto-clean \
  --dominance-ratio 2.0 \
  --force-overwrite
```

Outputs:
- `idea/datasets/features/auto_ngrams.json` - Auto-generated features
- `idea/datasets/features/manual_ngrams.json` - Curated features (edit for custom signals)
- `idea/datasets/features/overlap_analysis.json` - Shared vs unique features
- `idea/datasets/features/cleanup_report.json` - Dominance filtering report

**Underlying pipeline:**
1. `ngram_extractor.py` - Build per-tag n-gram counters (tri/four/five-grams)
2. `feature_analyzer.py` - Report shared vs discriminative features
3. `clean_ngrams.py` - Dominance-based filtering with safety net

### Tag Classifier Training

```bash
python idea/training/train_tag_classifier.py \
  --models idea/models/tag_classifier \
  --embedding-model intfloat/multilingual-e5-large-instruct \
  --epochs 50 --batch-size 64 --lr 3e-4 --patience 12 \
  --ngram-mode manual --force
```

**Process:**
- Loads `sts_train.csv`/`sts_eval.csv`, encodes tags via `LabelEncoder`
- Pulls n-gram features from `features/{manual,auto}_ngrams.json`
- Uses shared SentenceTransformer cache with classifier prefixes
- Builds hybrid architecture: embedding MLP + pattern MLP → fusion layer
- Stratified train/val split, early stopping, ReduceLROnPlateau
- Saves `unified_tag_classifier.pth` + metadata to `idea/models/tag_classifier/`

Pass `--force` to ignore fingerprint cache and retrain.

### FAISS Index Building

```bash
python idea/training/build_faiss_indices.py --global --force
```

**Process:**
- Loads `sts_train.csv` and `tag_to_answer.json`
- Encodes questions with STS passage prefix (`passage: `)
- Normalizes embeddings, builds FAISS Inner Product index
- Saves artifacts to `idea/models/semantic/`:
  - `faiss_index_global.index` - FAISS index
  - `sts_embeddings.npy` - Normalized embeddings
  - `question_mapping.csv` - Question→tag→answer mapping
  - `sts_metadata.json` - Fingerprint + prefix config

---

## Data Management

At runtime the matchers now look for artifacts beside the model directories first (e.g., `idea/models/semantic/../datasets/...`, `idea/models/tag_classifier/../datasets/...`) and then fall back to the repo defaults under `idea/datasets/`. This keeps answers and n-gram features aligned when you point `models_dir` to a custom location.

### Dataset Files

All first-party datasets now live under `idea/datasets/` (see `FaissMatcher.DATASETS_DIR` and
`TagClassifierMatcher.FEATURES_DIR`). Replace these files in place (or symlink them) if you
bring your own corpora, then rerun `artifacts.ensure_all()` so downstream fingerprints update.

| Path | Type | Consumed by | Notes |
| --- | --- | --- | --- |
| `idea/datasets/sts_train.csv` | CSV (`question,tag`) | `build_faiss_indices.py`, `train_tag_classifier.py`, `eval_pipeline.py` | Primary training corpus (36,358 committed rows) |
| `idea/datasets/sts_eval.csv` | CSV (`question,tag`) | `train_tag_classifier.py`, `eval_pipeline.py` | Held-out eval split (7,794 rows) |
| `idea/datasets/tag_to_answer.json` | JSON | `faiss_matcher.py`, `tag_classifier_matcher.py`, `IdeaPipeline` | Maps every tag to Bangla answer text bundled with releases |
| `idea/datasets/irrelevant.csv` | CSV (`question,tag,answer`) | Manual use | Curated chitchat/guardrail pairs for the `irrelevant` tag—append to `sts_train.csv` before rebuilding if you need it |
| `idea/datasets/features/manual_ngrams.json`* | JSON | `tag_classifier_matcher.py`, `train_tag_classifier.py` | Default n-gram pattern bank used at inference and training |
| `idea/datasets/features/auto_ngrams.json`* | JSON | `generate_features.py`, experiments | Auto-derived n-grams for analysis/ablation |
| `idea/datasets/features/overlap_analysis.json`* | JSON | Analysts | Diagnostics showing shared vs unique patterns per tag |
| `idea/datasets/features/cleanup_report.json`* | JSON | Analysts | Log of dominance filtering decisions |
| `idea/datasets/features/manual_ngrams_top40_backup.json`* | JSON | Safety net | Snapshot before manual pruning (handy when editing features) |

*Generated via `python idea/featurizer/generate_features.py --top-k 20 --auto-clean`.

### Artifact Outputs

`idea/utils/artifacts.ensure_all()` (or the individual training scripts) emits:

- **`idea/models/semantic/`**
  - `faiss_index_global.index` – FAISS inner-product index
  - `sts_embeddings.npy` – Normalized question embeddings
  - `question_mapping.csv` – Mirrors `sts_train.csv` with answer text for quick lookup
  - `sts_metadata.json` – Fingerprint + prefix/device metadata
- **`idea/models/tag_classifier/`**
  - `unified_tag_classifier.pth` – Torch weights for the dual-branch classifier
  - `unified_tag_classifier_metadata.json` – Fingerprint + normalization stats (`pattern_mean/std`, label encoder)

Run `python scripts/verify_setup.py` anytime to confirm these artifacts exist and to inspect their sizes.

### Adding New Data

```bash
# 1. Add questions to sts_train.csv (question,tag)
# 2. Add answers to tag_to_answer.json ({"tag": "answer"})

# 3. Rebuild artifacts
python -c "from idea.utils import artifacts; artifacts.ensure_all()"

# 4. Re-evaluate
python idea/evals/eval_pipeline.py --data idea/datasets/sts_eval.csv --top-k 1
```

`idea/datasets/irrelevant.csv` ships with small-talk/guardrail rows for an `irrelevant` tag.
Append those rows (or your own) into `sts_train.csv` and make sure `tag_to_answer.json` has
an `"irrelevant"` entry before rebuilding if you want the pipeline to surface that response.

### Data Utilities

**Deduplicate dataset:**
```bash
python idea/utils/data_utils/remove_duplicates.py \
  idea/datasets/sts_train.csv \
  idea/datasets/sts_train_dedup.csv
```

**Check for specific question/tag pairs:**
```bash
python idea/utils/data_utils/check_exact_matches.py \
  --train idea/datasets/sts_train.csv \
  --eval idea/datasets/sts_eval.csv \
  --questions "question1" "question2"
```

---

## Evaluation & Analysis

### Understanding Prediction Sources

The system can make predictions from three sources:

1. **`sts`** - FAISS semantic search only
2. **`classifier`** - Tag classifier only
3. **`sts+classifier`** - Fused from both signals (strongest)

**Why cosine similarity can be empty:**

When `prediction_source=classifier`, the classifier predicted a tag that wasn't in FAISS top-K results. This happens when:
- FAISS has poor recall for the question
- Classifier correctly identifies the tag using patterns/embeddings
- After normalization + weighting, classifier-only score beats all STS scores

**Example:**
```
input_question: "আমার অ্যাকাউন্ট লক কীভাবে খুলব?"
mapped_question: "" (empty - no good FAISS match)
expected_tag: account_locked_unlock_request
predicted_tag: account_locked_unlock_request
cosine_similarity: "" (empty - no STS match)
prediction_source: classifier
```

This demonstrates that:
- Classifier fills gaps where semantic search fails
- Dual redundancy is working correctly
- System handles paraphrased/unusual questions

### Analyzing Source Distribution

```bash
# Count prediction sources (requires simplified CSV)
cut -d',' -f6 idea/evals/results/eval_simple.csv | tail -n +2 | sort | uniq -c
```

**Healthy distribution:**
- **60-80% `sts+classifier`** - Both signals agree (strongest)
- **15-30% `classifier`** - Classifier filling FAISS gaps
- **0-10% `sts`** - STS alone

**Warning signs:**
- **>40% classifier-only** - FAISS recall poor, need better embeddings
- **>20% sts-only** - Classifier weak, needs retraining
- **<50% sts+classifier** - Signals disagreeing, check fusion weights

---

## Caching & Performance

### Artifact Fingerprinting

Training scripts (`train_tag_classifier.py`, `build_faiss_indices.py`) hash:
- CSV file sizes + modification times + contents
- Hyperparameters (learning rate, epochs, batch size)
- E5 prefix configuration

**Cache behavior:**
- Matching fingerprint → skip rebuild
- Changed inputs/config → rebuild required
- Use `--force` to bypass cache

### Shared Embedding Model

`idea/utils/model_cache.py` prevents redundant model loading:
- Caches `intfloat/multilingual-e5-large-instruct` (~1.2GB)
- Auto-selects device: CUDA → MPS → CPU
- Shared across FAISS, classifier, and evaluation scripts

### Auto-Build System

`idea/utils/artifacts.ensure_all()` orchestrates:
1. Check for missing features/classifier/FAISS artifacts
2. Generate n-gram features if needed
3. Train classifier if needed
4. Build FAISS index if needed

**Called by:**
- `idea/evals/eval_pipeline.py` (before evaluation)
- Interactive CLI (before first query)
- Manual invocation for fresh setups

---

## Troubleshooting

### E5 Prefix Changes Not Taking Effect

Any edit to `E5PrefixConfig` requires rebuilding ALL artifacts:
```bash
python idea/training/build_faiss_indices.py --global --force
python idea/training/train_tag_classifier.py \
  --models idea/models/tag_classifier --ngram-mode manual --force
```

### Slow Performance / Repeated Model Downloads

**Check cache:**
```bash
ls -lh ~/.cache/huggingface/hub/
df -h ~/.cache
```

Model should download once (~1.2GB) then cache. If loading repeatedly:
- Use long-lived Python processes
- Explicitly call `idea.utils.model_cache.get_shared_embedding_model()` once at startup
- Set `STS_EMBEDDING_DEVICE` explicitly

### CUDA Out of Memory / MPS Not Available

```bash
# Force CPU mode
export STS_EMBEDDING_DEVICE=cpu

# Or reduce batch size in training
# Edit: idea/training/train_tag_classifier.py
# Change: --batch-size 64 → --batch-size 32
```

### FileNotFoundError: sts_train.csv

**Cause:** Not running from repository root

**Fix:**
```bash
pwd  # Should end with /pipelineNLP
ls idea/datasets/sts_train.csv  # Should exist
```

### ensure_all() Hangs or Takes >2 Hours

**Check CPU usage:**
```bash
top  # Should show 100%+ during training
```

**Try manual steps to isolate:**
```bash
# Run each step individually
python idea/featurizer/generate_features.py --top-k 20 --auto-clean
python idea/training/build_faiss_indices.py --global --force
python idea/training/train_tag_classifier.py --force
```

---

## Development Workflows

### Experimenting with Ranker Weights

```python
# Edit idea/config.py
weights: Dict[str, float] = {
    "sts": 0.6,        # Increase STS weight
    "classifier": 0.4  # Decrease classifier weight
}
```

No rebuild needed - changes apply immediately at inference.

### Tuning Fusion Threshold

```python
# Edit idea/config.py → RankerConfig
min_score: float = 0.1  # Increase to filter more aggressively
```

Higher threshold = fewer but higher-confidence predictions.

### Debugging Score Issues

```python
from idea import IdeaPipeline

pipe = IdeaPipeline()
pipe.initialize()
response = pipe.run("test question", fusion_top_k=5)

# Inspect fusion process
print(response["telemetry"]["ranker"]["weights"])
print(response["telemetry"]["ranker"]["dropped_below_threshold"])

# See what was filtered
print(response["dropped_candidates"])

# Compare signal contributions
for candidate in response["candidates"]:
    print(f"{candidate['tag']}: {candidate['source']}")
```

### Adding Custom N-Gram Features

```bash
# 1. Generate initial features
python idea/featurizer/generate_features.py --top-k 20

# 2. Manually edit idea/datasets/features/manual_ngrams.json
# Add/remove n-grams for specific tags

# 3. Retrain classifier with manual features
python idea/training/train_tag_classifier.py --ngram-mode manual --force
```

---

## Command Reference

```bash
# Feature Generation
python idea/featurizer/generate_features.py --top-k 20 --auto-clean --dominance-ratio 2.0

# Training
python idea/training/train_tag_classifier.py --models idea/models/tag_classifier --ngram-mode manual --force
python idea/training/build_faiss_indices.py --global --force

# Testing
python idea/examples/interactive_cli.py --top-k 1

# Evaluation
python idea/evals/eval_pipeline.py --data idea/datasets/sts_eval.csv --top-k 1 --output-csv idea/evals/results/eval.csv
python idea/evals/simplify_csv.py idea/evals/results/eval.csv idea/evals/results/eval_simple.csv

# Data Utilities
python idea/utils/data_utils/remove_duplicates.py input.csv output.csv
python idea/utils/data_utils/check_exact_matches.py --train train.csv --eval eval.csv --questions "q1" "q2"

# Verification
python scripts/verify_setup.py
```

---

## Contributing

### Issues & Bugs

Report at: https://github.com/ehzawad/pipelineNLP/issues

### Architecture Notes for AI Assistants

This README consolidates guidance previously split across `CLAUDE.md` and `GETTING_STARTED.md`. Key design decisions:

1. **E5 prefix configuration is centralized** in `idea/config.py` → `E5PrefixConfig`
2. **Per-query min-max normalization** makes STS/classifier scores comparable
3. **Three independent top-K stages** (FAISS recall, classifier predictions, final fusion)
4. **Classifier-only predictions are valid** when FAISS has poor recall
5. **Shared embedding cache** prevents redundant 1.2GB model loads
6. **Fingerprint caching** prevents unnecessary retraining

When modifying the system:
- Prefix changes require rebuilding FAISS + classifier (`--force`)
- Weight/threshold changes apply immediately (no rebuild)
- N-gram feature changes require classifier retraining only

---


## Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) - Multilingual E5 embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [PyTorch](https://pytorch.org/) - Neural network training
