# -*- coding: utf-8 -*-
"""
Token Attention Classifier Training

Trains token-level attention models that process individual tokens BEFORE pooling
to preserve positional information needed for distinguishing similar tags.
"""

import json
import os
import logging
import time
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.utils.model_cache import get_shared_embedding_model
from idea.utils.token_utils import extract_token_embeddings, TOKEN_MAX_LENGTH
from idea.config import E5PrefixConfig, DEFAULT_EMBEDDING_MODEL
from idea.inference.token_attention_classifier import LightweightTokenClassifier, MultiHeadTokenClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Dataset for token-level attention classification"""

    def __init__(self, token_embeddings, attention_masks, labels):
        """
        Args:
            token_embeddings: (num_samples, max_seq_len, token_dim) tensor
            attention_masks: (num_samples, max_seq_len) boolean tensor
            labels: (num_samples,) array of label indices
        """
        self.token_embeddings = token_embeddings
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.token_embeddings[idx],  # (seq_len, token_dim)
            self.attention_masks[idx],   # (seq_len,)
            torch.LongTensor([self.labels[idx]])[0]
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.

    From: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()


class TokenAttentionTrainer:
    """Trainer for token-level attention classifiers"""

    def __init__(self, embedding_model=DEFAULT_EMBEDDING_MODEL, variant="lightweight"):
        """
        Args:
            embedding_model: Name of SentenceTransformer model
            variant: "lightweight" or "multihead"
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.device = None
        self.variant = variant
        self.tag_encoder = LabelEncoder()

    def load_data(self):
        """Load training/eval splits"""
        logger.info("Loading training and eval data...")

        component_dir = Path(__file__).parent.parent
        datasets_dir = component_dir / "datasets"

        # Load training data
        train_file = datasets_dir / "sts_train.csv"
        df_train = pd.read_csv(train_file)
        logger.info(f"  Train: {len(df_train)} questions (tags: {df_train['tag'].nunique()})")

        # Load eval data
        eval_file = datasets_dir / "sts_eval.csv"
        df_eval = pd.read_csv(eval_file)
        logger.info(f"  Eval: {len(df_eval)} questions (tags: {df_eval['tag'].nunique()})")

        return df_train, df_eval

    def prepare_embeddings(self, df, split_name="train", max_length: int = TOKEN_MAX_LENGTH):
        """Extract token-level embeddings for all questions"""
        logger.info(f"Extracting token embeddings for {split_name} set...")

        questions = df['question'].tolist()

        # Apply E5-instruct prefix
        questions_prefixed = E5PrefixConfig.format_classifier_queries_batch(questions)

        # Extract token embeddings (this returns tensors already)
        token_embeddings, attention_masks = extract_token_embeddings(
            model=self.embedding_model,
            texts=questions_prefixed,
            max_length=max_length,
            normalize=True,
            batch_size=32,
            show_progress=True
        )

        logger.info(f"  Token embeddings shape: {token_embeddings.shape}")
        logger.info(f"  Attention masks shape: {attention_masks.shape}")

        return token_embeddings, attention_masks

    def train_model(
        self,
        output_dir,
        epochs=50,
        batch_size=32,
        lr=0.0001,
        dropout=0.5,
        patience=10,
        use_focal_loss=True,
        focal_alpha=1.0,
        focal_gamma=2.0,
        hidden_dim=256,  # For lightweight variant
        num_heads=8,     # For multihead variant
        max_length: int = TOKEN_MAX_LENGTH,
        force=False
    ):
        """
        Train token attention classifier

        Args:
            output_dir: Where to save the trained model
            epochs: Maximum training epochs
            batch_size: Training batch size
            lr: Learning rate
            dropout: Dropout probability
            patience: Early stopping patience
            use_focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            hidden_dim: Hidden dimension for lightweight variant
            num_heads: Number of attention heads for multihead variant
            force: Force retrain even if model exists
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_file = output_dir / f"token_attention_{self.variant}.pth"
        if model_file.exists() and not force:
            logger.info(f"Model already exists at {model_file}. Use --force to retrain.")
            return

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)
        logger.info("Tokenization max_length: %d", max_length)
        token_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Load data
        df_train, df_eval = self.load_data()

        # Encode labels
        self.tag_encoder.fit(df_train['tag'])
        num_tags = len(self.tag_encoder.classes_)
        logger.info(f"Number of tags: {num_tags}")

        train_labels = self.tag_encoder.transform(df_train['tag'])
        eval_labels = self.tag_encoder.transform(df_eval['tag'])

        # Extract token embeddings
        train_token_embs, train_masks = self.prepare_embeddings(df_train, "train", max_length=max_length)
        eval_token_embs, eval_masks = self.prepare_embeddings(df_eval, "eval", max_length=max_length)

        # Create datasets
        train_dataset = TokenDataset(train_token_embs, train_masks, train_labels)
        eval_dataset = TokenDataset(eval_token_embs, eval_masks, eval_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        logger.info(f"Initializing {self.variant} token attention classifier")
        if self.variant == "lightweight":
            model = LightweightTokenClassifier(
                token_dim=token_dim,
                num_tags=num_tags,
                dropout=dropout,
                hidden_dim=hidden_dim
            ).to(self.device)
        else:  # multihead
            model = MultiHeadTokenClassifier(
                token_dim=token_dim,
                num_tags=num_tags,
                num_heads=num_heads,
                dropout=dropout
            ).to(self.device)

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Setup loss and optimizer
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using Cross Entropy Loss")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for token_embs, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                token_embs = token_embs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = model(token_embs, masks)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labels)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += len(labels)

            train_loss /= train_total
            train_acc = 100.0 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for token_embs, masks, labels in eval_loader:
                    token_embs = token_embs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)

                    logits = model(token_embs, masks)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * len(labels)
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += len(labels)

            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save best model
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'num_tags': num_tags,
                    'tag_encoder_classes': self.tag_encoder.classes_.tolist(),
                    'token_dim': token_dim,
                    'dropout': dropout,
                    'variant': self.variant,
                    'best_val_acc': best_val_acc,
                    'epoch': epoch + 1,
                    'e5_prefix_config': E5PrefixConfig.get_metadata(),
                }

                if self.variant == "lightweight":
                    checkpoint['hidden_dim'] = hidden_dim
                else:
                    checkpoint['num_heads'] = num_heads

                torch.save(checkpoint, model_file)
                logger.info(f"  ✓ Saved best model (val_acc={best_val_acc:.2f}%)")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        total_time = time.time() - train_start
        logger.info(f"\nTraining completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Save metadata
        metadata_file = output_dir / f"token_attention_{self.variant}_metadata.json"
        metadata = {
            'variant': self.variant,
            'embedding_model': self.embedding_model_name,
            'num_tags': num_tags,
            'token_dim': token_dim,
            'dropout': dropout,
            'best_val_acc': best_val_acc,
            'trained_at': datetime.now().isoformat(),
            'e5_prefix_config': E5PrefixConfig.get_metadata(),
        }

        if self.variant == "lightweight":
            metadata['hidden_dim'] = hidden_dim
        else:
            metadata['num_heads'] = num_heads

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to {metadata_file}")
