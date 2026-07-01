"""Tests for layered scene recognition helpers."""

import numpy as np
import torch

from learning.recognition import (
    cluster_spatial_instances,
    dbscan_points,
    fit_geometry,
    recognize_scene,
)
from simulation.generation import generate_sphere_surface


def test_dbscan_points_finds_two_clusters():
    first = np.random.randn(30, 3).astype(np.float32) * 0.02
    second = np.random.randn(30, 3).astype(np.float32) * 0.02 + np.array([2.0, 0.0, 0.0])
    labels = dbscan_points(np.vstack([first, second]), eps=0.1, min_points=5)
    assert set(labels) == {0, 1}


def test_cluster_spatial_instances_ignores_classes():
    points = np.vstack([
        np.random.randn(30, 3).astype(np.float32) * 0.02,
        np.random.randn(30, 3).astype(np.float32) * 0.02 + np.array([2.0, 0.0, 0.0]),
    ])
    instances = cluster_spatial_instances(points, eps=0.1, min_points=5)
    assert len(instances) == 2


def test_recognize_scene_assigns_cluster_labels():
    class FixedShapeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[4.0, 1.0, 0.0]], dtype=torch.float32).repeat(x.shape[0], 1)

    points = np.random.randn(30, 3).astype(np.float32) * 0.02
    pred_labels, instance_labels, instances = recognize_scene(
        points,
        FixedShapeModel(),
        device="cpu",
        cluster_eps=0.1,
        min_cluster_points=5,
        refine_geometry=False,
    )
    assert len(instances) == 1
    assert set(pred_labels.tolist()) == {0}
    assert set(instance_labels.tolist()) == {0}


def test_fit_geometry_prefers_sphere_for_sphere_points():
    np.random.seed(0)
    points = generate_sphere_surface(1, n_points=256, radius=1.0)[0]
    fit = fit_geometry(points, refine_label=True)
    assert fit.name == "sphere"
    assert fit.confidence > 0.8
