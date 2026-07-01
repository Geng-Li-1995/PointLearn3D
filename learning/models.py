"""PointNet-style model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, out_dim)
        self.mlp2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_points = x.shape[1]
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x.repeat(1, num_points, 1)


class PointFeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.local1 = LocalBlock(3, 64)
        self.local2 = LocalBlock(64, 128)
        self.local3 = LocalBlock(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local3(self.local2(self.local1(x)))


class PointSegmentationBackbone(nn.Module):
    def __init__(self, k_neighbors: int = 24):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(x)
        local_features = self.edge_mlp(self._edge_features(x)).max(dim=2)[0]
        point_features = self.fusion(torch.cat([point_features, local_features], dim=-1))
        global_features = torch.max(point_features, dim=1, keepdim=True)[0]
        global_features = global_features.repeat(1, x.shape[1], 1)
        return torch.cat([point_features, global_features], dim=-1)

    def _edge_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, dims = x.shape
        k = min(self.k_neighbors + 1, num_points)
        neighbor_idx = torch.topk(torch.cdist(x, x), k=k, largest=False).indices[:, :, 1:]
        flat_idx = neighbor_idx + torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
        neighbors = x.reshape(batch_size * num_points, dims)[flat_idx.reshape(-1)]
        neighbors = neighbors.reshape(batch_size, num_points, k - 1, dims)
        centers = x.unsqueeze(2).expand_as(neighbors)
        offsets = neighbors - centers
        distances = torch.norm(offsets, dim=-1, keepdim=True)
        return torch.cat([centers, offsets, distances], dim=-1)


class ShapeClassifier(nn.Module):
    """Classify an entire point cloud as cuboid / cylinder / sphere."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = PointFeatureBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(torch.max(features, dim=1)[0])


class SceneSegmenter(nn.Module):
    """Assign a shape label to every point in a multi-object scene."""

    def __init__(self, num_classes: int = 3, k_neighbors: int = 24):
        super().__init__()
        self.backbone = PointSegmentationBackbone(k_neighbors=k_neighbors)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
