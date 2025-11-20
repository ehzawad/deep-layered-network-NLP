"""
Token-level attention classifier that preserves positional information.

This architecture addresses the fundamental limitation of pooled embeddings by:
1. Processing individual tokens BEFORE pooling
2. Learning which tokens are important via attention
3. Preserving word order through positional encoding

Key innovation: Instead of using E5's pooled output (which loses position),
we use per-token embeddings and learn attention weights.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional encodings to preserve word order information.

    Based on "Attention Is All You Need" (Vaswani et al., 2017)

    Args:
        d_model: Dimension of token embeddings (default: 1024 for E5)
        max_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) - Token embeddings

        Returns:
            (batch, seq_len, d_model) - Token embeddings with positional info
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LightweightTokenClassifier(nn.Module):
    """
    Lightweight token-level classifier with learned attention weights.

    Architecture:
    1. Token embeddings (from E5) + positional encoding
    2. Learn scalar attention weight for each token
    3. Weighted pooling (instead of average pooling)
    4. Classification head

    This preserves positional information and learns which tokens are
    discriminative for classification.

    Args:
        token_dim: Dimension of token embeddings (default: 1024 for E5)
        num_tags: Number of output classes (default: 203)
        dropout: Dropout probability (default: 0.5)
        hidden_dim: Hidden dimension for attention (default: 256)
    """

    def __init__(
        self,
        token_dim=1024,
        num_tags=203,
        dropout=0.5,
        hidden_dim=256
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_tags = num_tags

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=token_dim,
            max_len=512,
            dropout=0.1
        )

        # Attention mechanism: Learn which tokens are important
        # Projects token to scalar attention score
        self.attention = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Scalar score per token
        )

        # Classification head: Process weighted representation
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tags)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, attention_mask=None, return_attention=False):
        """
        Forward pass with token-level attention.

        Args:
            token_embeddings: (batch, seq_len, token_dim) - E5 token embeddings
            attention_mask: (batch, seq_len) - Boolean mask (True = valid token)
            return_attention: If True, return attention weights for visualization

        Returns:
            logits: (batch, num_tags) - Classification scores
            attention_weights: (batch, seq_len) - Token importance (only if return_attention=True)
        """
        batch_size, seq_len, _ = token_embeddings.shape

        # Step 1: Add positional encoding (preserve word order)
        x = self.pos_encoding(token_embeddings)  # (batch, seq_len, token_dim)

        # Step 2: Compute attention scores for each token
        attn_scores = self.attention(x)  # (batch, seq_len, 1)

        # Step 3: Mask padding tokens (set their scores to -inf)
        if attention_mask is not None:
            # attention_mask: True = valid, False = padding
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(-1),  # (batch, seq_len, 1)
                float('-inf')
            )

        # Step 4: Softmax to get attention weights (sum to 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Step 5: Weighted pooling (instead of average pooling)
        # This gives higher weight to discriminative tokens
        weighted_repr = torch.sum(x * attn_weights, dim=1)  # (batch, token_dim)

        # Step 6: Classification
        logits = self.classifier(weighted_repr)  # (batch, num_tags)

        if return_attention:
            return logits, attn_weights.squeeze(-1)  # (batch, seq_len)

        return logits


class MultiHeadTokenClassifier(nn.Module):
    """
    Full multi-head attention classifier (heavier but more powerful).

    Uses self-attention to model token interactions before classification.
    This can capture relationships like "স্বামীর নাম" as a phrase.

    Args:
        token_dim: Dimension of token embeddings (default: 1024)
        num_tags: Number of output classes (default: 203)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        token_dim=1024,
        num_tags=203,
        num_heads=8,
        dropout=0.5
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_tags = num_tags

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=token_dim,
            max_len=512,
            dropout=0.1
        )

        # Self-attention: Let tokens interact with each other
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm after self-attention
        self.norm1 = nn.LayerNorm(token_dim)

        # Attention pooling: Learn which tokens to focus on
        self.attention_pooling = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tags)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, attention_mask=None, return_attention=False):
        """
        Forward pass with multi-head self-attention + attention pooling.

        Args:
            token_embeddings: (batch, seq_len, token_dim)
            attention_mask: (batch, seq_len) - Boolean mask
            return_attention: If True, return attention weights

        Returns:
            logits: (batch, num_tags)
            attention_weights: (batch, seq_len) - Only if return_attention=True
        """
        batch_size, seq_len, _ = token_embeddings.shape

        # Step 1: Positional encoding
        x = self.pos_encoding(token_embeddings)

        # Step 2: Self-attention (tokens interact with each other)
        # This captures: "স্বামীর" + "নাম" → "spouse name" (phrase)
        attn_output, _ = self.self_attention(
            x, x, x,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )

        # Residual connection + layer norm
        x = self.norm1(x + attn_output)

        # Step 3: Attention pooling (learn token importance)
        attn_scores = self.attention_pooling(x)  # (batch, seq_len, 1)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(-1),
                float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=1)

        # Step 4: Weighted pooling
        weighted_repr = torch.sum(x * attn_weights, dim=1)

        # Step 5: Classification
        logits = self.classifier(weighted_repr)

        if return_attention:
            return logits, attn_weights.squeeze(-1)

        return logits
