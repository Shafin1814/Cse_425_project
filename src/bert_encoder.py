"""
BERT encoder for music text understanding.
Wraps DistilBERT/BERT for multi-label tag classification.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERTEncoder(nn.Module):
    """BERT-based text encoder that outputs CLS embedding and full hidden states."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze_layers: int = 4,
        output_dim: int = 768,
    ):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim

        # Freeze early layers
        if hasattr(self.model, "transformer"):
            # DistilBERT
            layers = self.model.transformer.layer
        elif hasattr(self.model, "encoder"):
            # BERT
            layers = self.model.encoder.layer
        else:
            layers = []

        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze embeddings
        if hasattr(self.model, "embeddings"):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            dict with 'cls' (batch, hidden_dim) and 'hidden_states' (batch, seq_len, hidden_dim)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        cls_output = hidden_states[:, 0, :]  # CLS token

        return {
            "cls": cls_output,
            "hidden_states": hidden_states,
        }


class BERTTagClassifier(nn.Module):
    """
    Task 1: BERT-based multi-label tag classifier.
    Takes tokenized text and predicts binary tag presence.
    """

    def __init__(
        self,
        num_tags: int,
        model_name: str = "distilbert-base-uncased",
        freeze_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = BERTEncoder(model_name=model_name, freeze_layers=freeze_layers)
        hidden_dim = self.encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tags),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns: logits (batch, num_tags) — apply sigmoid for probabilities.
        """
        encoded = self.encoder(input_ids, attention_mask)
        logits = self.classifier(encoded["cls"])
        return logits


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Get the tokenizer for the BERT model."""
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)
