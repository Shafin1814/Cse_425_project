"""
Task 4: Contrastive dual-encoder for cross-modal MusicCaps alignment.
InfoNCE loss to learn shared embedding space between audio graphs and text captions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class ProjectionHead(nn.Module):
    """MLP projection head to shared embedding space."""

    def __init__(self, input_dim: int, projection_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveDualEncoder(nn.Module):
    """
    Task 4: Contrastive learning between GNN (audio graph) and BERT (caption).
    Uses InfoNCE loss with in-batch negatives.
    """

    def __init__(
        self,
        gnn_encoder,
        bert_encoder,
        graph_dim: int = 128,
        text_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.bert_encoder = bert_encoder
        self.temperature = temperature

        self.graph_projector = ProjectionHead(graph_dim, projection_dim)
        self.text_projector = ProjectionHead(text_dim, projection_dim)

    def encode_graph(self, graph_data: Data) -> torch.Tensor:
        """Encode graph to normalized projection."""
        g = self.gnn_encoder.encode_graph(graph_data)
        g_proj = self.graph_projector(g)
        return F.normalize(g_proj, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to normalized projection."""
        bert_out = self.bert_encoder(input_ids, attention_mask)
        t = bert_out["cls"]
        t_proj = self.text_projector(t)
        return F.normalize(t_proj, p=2, dim=-1)

    def forward(
        self,
        graph_data: Data,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Compute InfoNCE contrastive loss.

        Args:
            graph_data: batched PyG Data
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            dict with 'loss', 'graph_embeddings', 'text_embeddings', 'similarity'
        """
        g_emb = self.encode_graph(graph_data)     # (batch, proj_dim)
        t_emb = self.encode_text(input_ids, attention_mask)  # (batch, proj_dim)

        # Similarity matrix
        similarity = torch.matmul(g_emb, t_emb.t()) / self.temperature  # (batch, batch)

        # InfoNCE loss (symmetric)
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)

        loss_g2t = F.cross_entropy(similarity, labels)       # graph → text
        loss_t2g = F.cross_entropy(similarity.t(), labels)   # text → graph
        loss = (loss_g2t + loss_t2g) / 2

        return {
            "loss": loss,
            "graph_embeddings": g_emb,
            "text_embeddings": t_emb,
            "similarity": similarity,
        }

    @torch.no_grad()
    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k matches from gallery.

        Args:
            query_embeddings: (N, proj_dim) normalized embeddings
            gallery_embeddings: (M, proj_dim) normalized embeddings
            top_k: number of results to return

        Returns:
            scores: (N, top_k)
            indices: (N, top_k)
        """
        similarity = torch.matmul(query_embeddings, gallery_embeddings.t())
        scores, indices = similarity.topk(top_k, dim=-1)
        return scores, indices


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute Recall@K from a similarity matrix.

    Args:
        similarity_matrix: (N, N) — diagonal is the correct match
        ks: list of K values

    Returns:
        dict: {'R@1': float, 'R@5': float, 'R@10': float} for both directions
    """
    n = similarity_matrix.size(0)
    labels = torch.arange(n, device=similarity_matrix.device)

    metrics = {}
    k_max = min(max(ks), n)
    if k_max == 0:
        return {f"{d}_R@{k}": 0.0 for d in ["g2t", "t2g"] for k in ks}

    for direction, sim in [("g2t", similarity_matrix), ("t2g", similarity_matrix.t())]:
        _, indices = sim.topk(k_max, dim=-1)
        for k in ks:
            if k <= k_max:
                correct = (indices[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
                metrics[f"{direction}_R@{k}"] = correct.mean().item()
            else:
                correct = (indices == labels.unsqueeze(1)).any(dim=1).float()
                metrics[f"{direction}_R@{k}"] = correct.mean().item()

    return metrics
