"""
GNN-BERT Fusion model for multi-context understanding (Task 3).
Implements cross-attention and concatenation fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import Data


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism: graph embedding queries text hidden states.
    Q = g @ W_Q, K = H_text @ W_K, V = H_text @ W_V
    """

    def __init__(self, graph_dim: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert text_dim % num_heads == 0, "text_dim must be divisible by num_heads"

        self.W_Q = nn.Linear(graph_dim, text_dim)
        self.W_K = nn.Linear(text_dim, text_dim)
        self.W_V = nn.Linear(text_dim, text_dim)
        self.W_O = nn.Linear(text_dim, text_dim)

        self.layer_norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        graph_embedding: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            graph_embedding: (batch, graph_dim)
            text_hidden_states: (batch, seq_len, text_dim)
            text_attention_mask: (batch, seq_len)

        Returns:
            attended: (batch, text_dim)
        """
        batch_size = graph_embedding.size(0)
        seq_len = text_hidden_states.size(1)

        # Project graph embedding to query (expand to seq dimension)
        Q = self.W_Q(graph_embedding).unsqueeze(1)  # (batch, 1, text_dim)
        K = self.W_K(text_hidden_states)  # (batch, seq_len, text_dim)
        V = self.W_V(text_hidden_states)  # (batch, seq_len, text_dim)

        # Multi-head reshape
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if text_attention_mask is not None:
            mask = text_attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)  # (batch, heads, 1, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1)  # (batch, text_dim)
        attended = self.W_O(attended)
        attended = self.layer_norm(attended)

        return attended


class GNNBERTFusionModel(nn.Module):
    """
    Task 3: Fuses GNN graph embedding with BERT text embedding.
    Supports cross-attention and simple concatenation fusion.
    """

    def __init__(
        self,
        gnn_encoder,
        bert_encoder,
        num_tags: int,
        fusion_method: str = "cross_attention",
        graph_dim: int = 128,
        text_dim: int = 768,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.bert_encoder = bert_encoder
        self.fusion_method = fusion_method

        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(graph_dim, text_dim, num_heads=4)
            # Fused dim = graph_dim + text_dim (concat graph + attended text)
            fused_dim = graph_dim + text_dim
        elif fusion_method == "concat":
            fused_dim = graph_dim + text_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_tags),
        )

    def forward(
        self,
        graph_data: Data,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            graph_data: PyG Data with x, edge_index, batch
            input_ids: (batch, seq_len) BERT token IDs
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_tags)
        """
        # Encode graph
        g = self.gnn_encoder.encode_graph(graph_data)  # (batch, graph_dim)

        # Encode text
        bert_out = self.bert_encoder(input_ids, attention_mask)
        t = bert_out["cls"]  # (batch, text_dim)
        h_text = bert_out["hidden_states"]  # (batch, seq_len, text_dim)

        # Fusion
        if self.fusion_method == "cross_attention":
            attended = self.fusion(g, h_text, attention_mask)  # (batch, text_dim)
            z = torch.cat([g, attended], dim=-1)  # (batch, graph_dim + text_dim)
        elif self.fusion_method == "concat":
            z = torch.cat([g, t], dim=-1)  # (batch, graph_dim + text_dim)

        logits = self.classifier(z)
        return logits

    def get_fused_embedding(
        self,
        graph_data: Data,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the fused embedding z for visualization (t-SNE)."""
        g = self.gnn_encoder.encode_graph(graph_data)
        bert_out = self.bert_encoder(input_ids, attention_mask)
        t = bert_out["cls"]
        h_text = bert_out["hidden_states"]

        if self.fusion_method == "cross_attention":
            attended = self.fusion(g, h_text, attention_mask)
            z = torch.cat([g, attended], dim=-1)
        elif self.fusion_method == "concat":
            z = torch.cat([g, t], dim=-1)

        return z
