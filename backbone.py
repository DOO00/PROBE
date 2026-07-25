from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class _BaseTwoLayerGNN(nn.Module):
    """Two-layer GNN backbone with an explicit classifier head.

    The forward method returns:
    1) logits: [num_nodes, num_classes]
    2) embeddings: penultimate features before classifier, [num_nodes, embed_channels]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        embed_channels: int | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        embed_channels = hidden_channels if embed_channels is None else embed_channels
        self.dropout = dropout
        self.conv1 = self._build_conv(in_channels, hidden_channels)
        self.conv2 = self._build_conv(hidden_channels, embed_channels)
        self.classifier = nn.Linear(embed_channels, num_classes)

    def _build_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, x, edge_index=None):
        # Support both forward(x, edge_index) and forward(data) styles.
        if edge_index is None:
            if not (hasattr(x, "x") and hasattr(x, "edge_index")):
                raise ValueError("Expected (x, edge_index) or a PyG data object with x/edge_index.")
            data = x
            x, edge_index = data.x, data.edge_index

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        embeddings = F.relu(h)

        logits = self.classifier(F.dropout(embeddings, p=self.dropout, training=self.training))
        return logits, embeddings


class TwoLayerGraphSAGE(_BaseTwoLayerGNN):
    """Two-layer GraphSAGE backbone."""

    def _build_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        return SAGEConv(in_channels, out_channels)


class TwoLayerGCN(_BaseTwoLayerGNN):
    """Two-layer GCN backbone."""

    def _build_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        return GCNConv(in_channels, out_channels)


def build_backbone(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    num_classes: int,
    embed_channels: int | None = None,
    dropout: float = 0.5,
) -> nn.Module:
    """Factory function for two-layer PyG backbones.

    Args:
        model_name: "sage" / "graphsage" or "gcn".
    """
    model_name = model_name.lower()
    if model_name in {"sage", "graphsage"}:
        return TwoLayerGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            embed_channels=embed_channels,
            dropout=dropout,
        )
    if model_name == "gcn":
        return TwoLayerGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            embed_channels=embed_channels,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_name: {model_name}. Use 'sage'/'graphsage' or 'gcn'.")

