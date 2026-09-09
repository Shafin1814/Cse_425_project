"""
GNN models for music structure understanding.
Implements GraphSAGE and GAT for genre/tag classification.
Also includes a CNN mel-spectrogram baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data


class GraphSAGEModel(nn.Module):
    """
    Task 2: GraphSAGE encoder for music segment graphs.
    Performs graph-level classification via mean pooling.
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dim: int = 128,
        output_dim: int = 10,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.graph_dim = hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_graph(self, data: Data) -> torch.Tensor:
        """
        Encode a graph to a graph-level embedding.

        Args:
            data: PyG Data object with x, edge_index, batch

        Returns:
            Graph-level embedding (batch_size, hidden_dim)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        g = global_mean_pool(x, batch)
        return g

    def forward(self, data: Data) -> torch.Tensor:
        """
        Returns: logits (batch_size, output_dim)
        """
        g = self.encode_graph(data)
        logits = self.classifier(g)
        return logits


class GATModel(nn.Module):
    """
    Alternative GNN: Graph Attention Network for music graphs.
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dim: int = 128,
        output_dim: int = 10,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.graph_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_graph(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)
        return g

    def forward(self, data: Data) -> torch.Tensor:
        g = self.encode_graph(data)
        logits = self.classifier(g)
        return logits


class CNNMelBaseline(nn.Module):
    """
    Baseline: CNN on mel-spectrograms for genre classification.
    Input: (batch, 1, n_mels, time_steps)
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time_steps)
        Returns:
            logits (batch, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
