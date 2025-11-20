"""Unified tag classifier architecture (embedding + pattern branches with FiLM conditioning)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedTagClassifier(nn.Module):
    """
    Two-branch architecture (embedding + pattern) with FiLM conditioning and deeper fusion.

    Architecture:
    - Embedding branch: 1024 → 512 → 256
    - Pattern branch: 1015 → 192 → 128 (WIDER: was 64)
    - FiLM: Pattern generates scale (gamma) and shift (beta) to condition embeddings
    - Fusion: Deeper path with 384 → 256 → 192 → 128 → 203
    - Auxiliary heads: Each branch can classify independently (training only)

    Args:
        embedding_dim: Dimension of embedding input (default: 1024 for E5-large)
        pattern_dim: Dimension of n-gram pattern features (default: 1015)
        num_tags: Number of output classes (default: 203)
        dropout: Dropout probability (default: 0.5)
        pattern_output_dim: Output dimension of pattern branch (default: 128, was 64)
        use_film: Enable FiLM conditioning (default: True)
        use_auxiliary: Enable auxiliary classification heads (default: True)
    """

    def __init__(
        self,
        embedding_dim=1024,
        pattern_dim=1015,
        num_tags=203,
        dropout=0.5,
        pattern_output_dim=128,
        use_film=True,
        use_auxiliary=True
    ):
        super().__init__()

        self.use_film = use_film
        self.use_auxiliary = use_auxiliary
        self.pattern_output_dim = pattern_output_dim
        self.embedding_output_dim = 256

        # Embedding branch
        self.emb_fc1 = nn.Linear(embedding_dim, 512)
        self.emb_bn1 = nn.BatchNorm1d(512)
        self.emb_fc2 = nn.Linear(512, self.embedding_output_dim)
        self.emb_bn2 = nn.BatchNorm1d(self.embedding_output_dim)

        # Pattern branch (WIDER: 64 → 128)
        self.pattern_fc1 = nn.Linear(pattern_dim, 192)
        self.pattern_bn1 = nn.BatchNorm1d(192)
        self.pattern_fc2 = nn.Linear(192, pattern_output_dim)
        self.pattern_bn2 = nn.BatchNorm1d(pattern_output_dim)

        # FiLM conditioning: Pattern branch generates scale & shift for embeddings
        if self.use_film:
            self.film_gamma = nn.Linear(pattern_output_dim, self.embedding_output_dim)
            self.film_beta = nn.Linear(pattern_output_dim, self.embedding_output_dim)
            self.film_bn = nn.BatchNorm1d(self.embedding_output_dim)
            # Initialize gamma to near 1.0 (pass-through) and beta to 0
            nn.init.normal_(self.film_gamma.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.film_gamma.bias, 1.0)
            nn.init.normal_(self.film_beta.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.film_beta.bias, 0.0)

        # Fusion (DEEPER: 3 hidden layers instead of 1)
        # Input: 384-dim (256 emb + 128 pat) when FiLM enabled
        #        320-dim (256 emb + 64 pat) when FiLM disabled (backward compat)
        fusion_input_dim = self.embedding_output_dim + pattern_output_dim
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_fc2 = nn.Linear(256, 192)
        self.fusion_bn2 = nn.BatchNorm1d(192)
        self.fusion_fc3 = nn.Linear(192, 128)
        self.fusion_bn3 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, num_tags)

        # Auxiliary classification heads (force each branch to learn discriminative features)
        if self.use_auxiliary:
            self.aux_emb_classifier = nn.Linear(self.embedding_output_dim, num_tags)
            self.aux_pat_classifier = nn.Linear(pattern_output_dim, num_tags)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, patterns, return_aux=False):
        """
        Forward pass with optional auxiliary outputs.

        Args:
            embeddings: (batch, embedding_dim) - E5 embeddings
            patterns: (batch, pattern_dim) - N-gram pattern features
            return_aux: If True, return (main_logits, aux_emb_logits, aux_pat_logits)
                       If False, return main_logits only

        Returns:
            main_logits: (batch, num_tags) - Main classification logits
            aux_emb_logits: (batch, num_tags) - Embedding branch logits (only if return_aux=True)
            aux_pat_logits: (batch, num_tags) - Pattern branch logits (only if return_aux=True)
        """
        # Embedding branch
        emb = F.relu(self.emb_bn1(self.emb_fc1(embeddings)))
        emb = self.dropout(emb)
        emb = F.relu(self.emb_bn2(self.emb_fc2(emb)))  # (batch, 256)
        emb = self.dropout(emb)

        # Pattern branch (WIDER: 128 instead of 64)
        pat = F.relu(self.pattern_bn1(self.pattern_fc1(patterns)))
        pat = self.dropout(pat)
        pat = F.relu(self.pattern_bn2(self.pattern_fc2(pat)))  # (batch, 128)
        pat = self.dropout(pat)

        # FiLM conditioning: Pattern branch modulates embedding features
        if self.use_film:
            gamma = self.film_gamma(pat)  # (batch, 256) - scale
            beta = self.film_beta(pat)    # (batch, 256) - shift

            # Affine transformation: emb_conditioned = gamma * emb + beta
            # This allows pattern branch to:
            # - Suppress irrelevant embedding dimensions (gamma ≈ 0)
            # - Amplify relevant dimensions (gamma > 1)
            # - Add bias based on detected patterns (beta)
            emb_conditioned = self.film_bn(gamma * emb + beta)
            emb_for_fusion = emb_conditioned
        else:
            emb_for_fusion = emb

        # Fusion (DEEPER: 3 hidden layers)
        combined = torch.cat([emb_for_fusion, pat], dim=1)  # (batch, 384 or 320)

        fused = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        fused = self.dropout(fused)
        fused = F.relu(self.fusion_bn2(self.fusion_fc2(fused)))
        fused = self.dropout(fused)
        fused = F.relu(self.fusion_bn3(self.fusion_fc3(fused)))
        fused = self.dropout(fused)

        # Main output
        main_logits = self.output(fused)

        # Auxiliary outputs (training only)
        if return_aux and self.use_auxiliary:
            aux_emb_logits = self.aux_emb_classifier(emb)
            aux_pat_logits = self.aux_pat_classifier(pat)
            return main_logits, aux_emb_logits, aux_pat_logits

        return main_logits
