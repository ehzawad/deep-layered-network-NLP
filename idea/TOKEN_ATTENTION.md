# Token-Level Attention Classifier

## Overview

This document describes the token-level attention architecture implemented to address the fundamental limitation of pooled embeddings: **loss of positional information**.

## The Problem

### Why FiLM Failed

The previous FiLM (Feature-wise Linear Modulation) approach didn't improve accuracy because:

1. **E5 pooling happens BEFORE we can use positional information**
   - `model.encode()` returns a single 1024-dim vector per sentence
   - Word order is already lost by the time FiLM modulates features

2. **N-gram features are bag-of-words**
   - Counting "স্বামীর" matches doesn't tell us WHERE it appears in the sentence
   - "আমার স্বামীর নাম ঠিক করতে" vs "নাম ঠিক করতে আমার স্বামীর" look identical

3. **Syntactic similarity bias**
   - Tags differ by 1-2 words (e.g., "spouse correction" vs "parent correction")
   - Same sentence structure but different intent
   - Need to know which tokens are discriminative

## The Solution: Token Attention

### Key Innovation

**Process individual tokens BEFORE pooling** to preserve positional information.

Instead of:
```
Text → E5 Encoder → Pooled Embedding (1024-dim) → Classifier
                    ↑ Position info lost here
```

We do:
```
Text → E5 Encoder → Token Embeddings (seq_len × 1024) → Attention → Classifier
                                                        ↑ Learn which tokens matter
```

## Architecture

### Two Variants

#### 1. LightweightTokenClassifier (~2M parameters)

**Recommended for production** - fast and efficient.

```python
Architecture:
1. Positional Encoding (sinusoidal, preserves word order)
2. Attention Mechanism (token_dim → hidden_dim → scalar score)
3. Weighted Pooling (instead of average pooling)
4. Classification Head (token_dim → 512 → 256 → num_tags)

Forward Pass:
- Input: (batch, seq_len, 1024) token embeddings
- Add positional encoding
- Compute attention score for each token
- Softmax to get attention weights (sum to 1)
- Weighted pooling: weighted_repr = Σ(token * attention_weight)
- Classify weighted representation
```

**Key benefit**: Learns which tokens are discriminative (e.g., "স্বামীর" gets high attention for spouse-related tags).

#### 2. MultiHeadTokenClassifier (~8M parameters)

**For maximum accuracy** - heavier but more powerful.

```python
Architecture:
1. Positional Encoding
2. Multi-Head Self-Attention (tokens interact with each other)
3. Layer Normalization + Residual Connection
4. Attention Pooling (learns token importance)
5. Classification Head

Forward Pass:
- Input: (batch, seq_len, 1024) token embeddings
- Add positional encoding
- Self-attention: tokens interact (captures phrases like "স্বামীর নাম")
- Residual + LayerNorm
- Attention pooling to get weighted representation
- Classify weighted representation
```

**Key benefit**: Captures token interactions (phrase-level semantics).

## Implementation Files

### Core Architecture

1. **`idea/inference/token_attention_classifier.py`**
   - `PositionalEncoding`: Sinusoidal positional encodings
   - `LightweightTokenClassifier`: Simple attention-based pooling
   - `MultiHeadTokenClassifier`: Full self-attention

2. **`idea/utils/token_utils.py`**
   - `extract_token_embeddings()`: Extracts per-token embeddings from SentenceTransformer
   - `extract_token_embeddings_numpy()`: Same but returns numpy arrays
   - `visualize_token_attention()`: Visualize which tokens got high attention

### Training

3. **`idea/training/token_attention_trainer.py`**
   - `TokenAttentionTrainer`: Trainer class for token attention models
   - `TokenDataset`: Dataset that handles token sequences
   - `FocalLoss`: Focal loss for class imbalance

4. **`idea/training/train_token_attention.py`**
   - CLI entry point for training

### Inference

5. **`idea/inference/tag_classifier_matcher.py`** (modified)
   - Now supports both legacy and token attention models
   - Automatically selects the right inference path based on config

6. **`idea/config.py`** (modified)
   - Added `use_token_attention` and `token_attention_variant` to `TagClassifierConfig`

### Testing

7. **`idea/examples/test_token_attention.py`**
   - Standalone test script
   - Can train, test inference, and compare legacy vs token attention

## Usage

### Training

```bash
# Train lightweight variant (recommended, ~2M params)
python idea/training/train_token_attention.py --variant lightweight

# Train multihead variant (heavier, ~8M params)
python idea/training/train_token_attention.py --variant multihead

# Custom hyperparameters
python idea/training/train_token_attention.py \
    --variant lightweight \
    --epochs 100 \
    --batch-size 48 \
    --lr 0.0001 \
    --dropout 0.5 \
    --hidden-dim 256 \
    --use-focal-loss \
    --force
```

### Inference

#### Option 1: Using the test script

```bash
# Test inference only
python idea/examples/test_token_attention.py --test --variant lightweight

# Compare legacy vs token attention side-by-side
python idea/examples/test_token_attention.py --compare
```

#### Option 2: Programmatic usage

```python
from idea.config import TagClassifierConfig
from idea.inference.tag_classifier_matcher import TagClassifierMatcher

# Initialize with token attention
config = TagClassifierConfig(
    use_token_attention=True,
    token_attention_variant="lightweight"  # or "multihead"
)
matcher = TagClassifierMatcher(config)
matcher.initialize()

# Run inference
result = matcher.predict("আমার স্বামীর নাম ঠিক করতে কি লাগবে?", top_k=3)
for pred in result['results']:
    print(f"{pred['tag']} (confidence: {pred['confidence']:.4f})")
```

#### Option 3: Through the pipeline

The existing `IdeaPipeline` and `interactive_cli.py` will automatically use token attention if configured:

```python
from idea import IdeaPipeline
from idea.config import IdeaConfig, TagClassifierConfig

# Configure pipeline to use token attention
config = IdeaConfig(
    classifier=TagClassifierConfig(
        use_token_attention=True,
        token_attention_variant="lightweight"
    )
)

pipe = IdeaPipeline(config)
pipe.initialize()
result = pipe.run("আমার স্বামীর নাম ঠিক করতে কি লাগবে?")
```

## Expected Improvements

### Accuracy Gains

**Realistic expectation: +2-4% accuracy improvement**

From current 91.7% → target 93.5-95.5%

### Why This Should Work

1. **Preserves word order**: Positional encoding ensures the model knows token positions
2. **Learns discriminative tokens**: Attention mechanism learns which words matter
3. **Handles syntactic similarity**: Can distinguish "স্বামীর নাম" from "আব্বার নাম" even with identical structure

### What Won't Improve

- Data quality issues (mislabeled examples)
- Genuinely ambiguous questions
- Tags that truly have no distinguishing features

## Model Checkpoints

Token attention models are saved with a different naming convention:

```
idea/models/tag_classifier/
├── token_attention_lightweight.pth          # Lightweight model checkpoint
├── token_attention_lightweight_metadata.json
├── token_attention_multihead.pth            # Multihead model checkpoint
├── token_attention_multihead_metadata.json
├── unified_tag_classifier.pth               # Legacy model (still works)
└── unified_tag_classifier_metadata.json
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_tags': 203,
    'tag_encoder_classes': ['tag1', 'tag2', ...],  # Tag names in order
    'token_dim': 1024,
    'dropout': 0.5,
    'variant': 'lightweight',  # or 'multihead'
    'best_val_acc': 94.23,
    'epoch': 42,
    'hidden_dim': 256,  # For lightweight
    # OR
    'num_heads': 8,     # For multihead
    'e5_prefix_config': {...}
}
```

## Technical Details

### How Token Embeddings Are Extracted

```python
# Instead of using pooled output:
embedding = model.encode([text])  # Shape: (1024,) - position info lost

# We extract token embeddings:
transformer_model = model[0].auto_model  # Access underlying transformer
outputs = transformer_model(input_ids, attention_mask)
token_embeddings = outputs.last_hidden_state  # Shape: (seq_len, 1024)
```

### Positional Encoding Formula

Based on "Attention Is All You Need" (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
- pos = token position (0, 1, 2, ...)
- i = dimension index (0 to d_model/2)
- d_model = 1024 (E5 embedding dimension)
```

### Attention Mechanism

For lightweight variant:

```python
# 1. Project token to scalar score
attention_scores = Linear(token_embeddings)  # (seq_len, 1)

# 2. Mask padding tokens
attention_scores[padding] = -inf

# 3. Softmax to get weights
attention_weights = softmax(attention_scores)  # (seq_len, 1), sum to 1

# 4. Weighted pooling
weighted_repr = sum(token_embeddings * attention_weights)  # (1024,)
```

For multihead variant, same but with multi-head self-attention first.

## Training Details

### Hyperparameters

| Parameter | Lightweight | Multihead | Notes |
|-----------|-------------|-----------|-------|
| Epochs | 50 | 50 | Early stopping with patience=10 |
| Batch Size | 32 | 32 | Fits in 3GB VRAM |
| Learning Rate | 0.0001 | 0.0001 | Adam optimizer |
| Dropout | 0.5 | 0.5 | Regularization |
| Hidden Dim | 256 | N/A | Attention projection dim |
| Num Heads | N/A | 8 | Self-attention heads |
| Loss | Focal Loss | Focal Loss | alpha=1.0, gamma=2.0 |

### Training Data

- **Train**: 36,358 questions, 203 tags
- **Eval**: 7,794 questions, 203 tags
- **E5 Model**: `intfloat/multilingual-e5-large-instruct`
- **Prefix**: `"query: "` (E5-instruct format)

### Training Time

- **Lightweight**: ~10-15 minutes per epoch on GPU
- **Multihead**: ~20-25 minutes per epoch on GPU

## Comparison: Legacy vs Token Attention

| Feature | Legacy Unified Classifier | Token Attention |
|---------|---------------------------|-----------------|
| Input | Pooled embedding (1024-dim) | Token sequences (seq_len × 1024) |
| Positional Info | ❌ Lost after pooling | ✅ Preserved via positional encoding |
| N-gram Features | ✅ Required | ❌ Not needed |
| Parameters | ~2M (with FiLM) | ~2M (lightweight) / ~8M (multihead) |
| Inference Speed | Fast | Slightly slower (processes tokens) |
| Accuracy | 91.7% | Expected: 93.5-95.5% |
| Memory | Low | Medium (stores token sequences) |

## Backward Compatibility

The implementation is fully backward compatible:

1. Legacy models still work (controlled by `use_token_attention=False`)
2. Existing inference code unchanged
3. No changes needed to `IdeaPipeline` or `interactive_cli.py`
4. Both model types can coexist in the same directory

## Troubleshooting

### Model Not Found

```
FileNotFoundError: Classifier model not found at idea/models/tag_classifier/token_attention_lightweight.pth
```

**Solution**: Train the model first:
```bash
python idea/training/train_token_attention.py --variant lightweight --force
```

### Out of Memory

If you get CUDA out of memory errors during training:

1. Reduce batch size: `--batch-size 16`
2. Use lightweight variant instead of multihead
3. Reduce max sequence length in `token_utils.py` (currently 512)

### Slow Inference

Token attention is slightly slower than legacy because it processes token sequences:

- **Legacy**: ~10ms per question
- **Token attention**: ~20-30ms per question (still fast enough for production)

If speed is critical, use lightweight variant or stick with legacy model.

## Future Improvements

Potential enhancements (not implemented yet):

1. **Token-level n-gram features**: Combine token attention with positional n-gram matching
2. **Hierarchical attention**: Sentence-level + token-level attention
3. **Distillation**: Train a smaller student model from the multihead teacher
4. **Adaptive pooling**: Learn pooling strategy instead of fixed attention

## References

- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- "Focal Loss for Dense Object Detection" (Lin et al., 2017) - Focal loss
- E5 Embeddings: https://huggingface.co/intfloat/multilingual-e5-large-instruct
