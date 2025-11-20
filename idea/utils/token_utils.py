"""
Utility functions for extracting token-level embeddings from SentenceTransformers.

This module provides helpers to get per-token embeddings (before pooling)
from E5 models, preserving positional information needed for attention mechanisms.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Shared sequence length for token-level models to balance speed/coverage.
TOKEN_MAX_LENGTH = 256

def extract_token_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    max_length: int = TOKEN_MAX_LENGTH,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract token-level embeddings from SentenceTransformer model.

    Instead of using the pooled output (which loses positional information),
    this extracts per-token embeddings from the transformer layers.

    Args:
        model: SentenceTransformer instance (e.g., E5-large)
        texts: List of input texts to encode
        max_length: Maximum sequence length (default: 256)
        normalize: Whether to L2-normalize embeddings (default: True)
        batch_size: Batch size for encoding (default: 32)
        show_progress: Show progress bar (default: False)

    Returns:
        token_embeddings: (num_texts, max_seq_len, hidden_dim)
                         Padded token embeddings (padded tokens are zeros)
        attention_mask: (num_texts, max_seq_len)
                       Boolean mask (True = valid token, False = padding)

    Example:
        >>> model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        >>> texts = ["আমার স্বামীর নাম ঠিক করতে কি লাগবে?"]
        >>> token_embs, mask = extract_token_embeddings(model, texts)
        >>> print(token_embs.shape)  # (1, seq_len, 1024)
    """
    device = model.device
    tokenizer = model.tokenizer

    # Tokenize all texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        return_attention_mask=True
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Get the underlying transformer model
    # For E5, the structure is: SentenceTransformer -> [transformer modules] -> model
    # We need to access the actual transformer (usually first module)
    transformer_model = model[0].auto_model  # Access the transformer

    all_token_embeddings = []

    # Process in batches
    num_texts = len(texts)

    if show_progress:
        from tqdm import tqdm
        batch_iterator = tqdm(
            range(0, num_texts, batch_size),
            desc="Extracting token embeddings",
            total=(num_texts + batch_size - 1) // batch_size
        )
    else:
        batch_iterator = range(0, num_texts, batch_size)

    for i in batch_iterator:
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]

        with torch.no_grad():
            # Get transformer outputs
            outputs = transformer_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                return_dict=True
            )

            # Extract token embeddings (last hidden state)
            # Shape: (batch_size, seq_len, hidden_dim)
            token_embeddings = outputs.last_hidden_state

            # Optional: Normalize token embeddings
            if normalize:
                token_embeddings = torch.nn.functional.normalize(
                    token_embeddings,
                    p=2,
                    dim=-1
                )

            all_token_embeddings.append(token_embeddings.cpu())

    # Concatenate all batches
    token_embeddings = torch.cat(all_token_embeddings, dim=0)

    # Convert attention mask to boolean (True = valid, False = padding)
    attention_mask_bool = attention_mask.cpu().bool()

    return token_embeddings, attention_mask_bool


def extract_token_embeddings_numpy(
    model: SentenceTransformer,
    texts: List[str],
    max_length: int = TOKEN_MAX_LENGTH,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract token-level embeddings and return as numpy arrays.

    Same as extract_token_embeddings but returns numpy arrays instead of tensors.

    Args:
        model: SentenceTransformer instance
        texts: List of input texts
        max_length: Maximum sequence length
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for encoding
        show_progress: Show progress bar

    Returns:
        token_embeddings: (num_texts, max_seq_len, hidden_dim) numpy array
        attention_mask: (num_texts, max_seq_len) boolean numpy array
    """
    token_embeddings, attention_mask = extract_token_embeddings(
        model=model,
        texts=texts,
        max_length=max_length,
        normalize=normalize,
        batch_size=batch_size,
        show_progress=show_progress
    )

    return token_embeddings.numpy(), attention_mask.numpy()


def visualize_token_attention(
    text: str,
    tokens: List[str],
    attention_weights: np.ndarray,
    top_k: int = 10
) -> str:
    """
    Visualize which tokens received high attention scores.

    Args:
        text: Original input text
        tokens: List of tokens (from tokenizer)
        attention_weights: (seq_len,) attention scores
        top_k: Number of top tokens to show

    Returns:
        String representation of attention visualization

    Example:
        >>> text = "আমার স্বামীর নাম ঠিক করতে কি লাগবে?"
        >>> tokens = ["আমার", "স্বামীর", "নাম", "ঠিক", "করতে", "কি", "লাগবে", "?"]
        >>> attention = np.array([0.05, 0.82, 0.65, 0.12, 0.08, 0.04, 0.03, 0.01])
        >>> print(visualize_token_attention(text, tokens, attention))
        Top 10 tokens by attention:
          1. স্বামীর (0.820) ████████████████
          2. নাম     (0.650) █████████████
          3. ঠিক    (0.120) ██
          ...
    """
    # Sort tokens by attention weight
    token_attention = list(zip(tokens, attention_weights))
    token_attention.sort(key=lambda x: x[1], reverse=True)

    # Build visualization string
    lines = ["Top {} tokens by attention:".format(min(top_k, len(tokens)))]

    for i, (token, weight) in enumerate(token_attention[:top_k], 1):
        # Create bar visualization
        bar_length = int(weight * 20)  # Scale to 20 characters max
        bar = "█" * bar_length

        lines.append(f"  {i}. {token:12s} ({weight:.3f}) {bar}")

    return "\n".join(lines)
