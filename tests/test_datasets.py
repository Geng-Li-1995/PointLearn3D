"""Tests for PyTorch datasets (on-the-fly mode, no disk cache)."""

import torch

from learning.datasets import SceneDataset, ShapeDataset


def test_shape_dataset_item_shapes():
    ds = ShapeDataset(num_samples=4, num_points=128, cache=False, seed=0)
    points, label = ds[0]
    assert points.shape == (128, 3)
    assert label.dtype == torch.long
    assert 0 <= label.item() < 3


def test_scene_dataset_item_shapes():
    ds = SceneDataset(num_samples=2, num_points=256, cache=False, seed=0)
    points, labels = ds[0]
    assert points.shape == (256, 3)
    assert labels.shape == (256,)
    assert labels.dtype == torch.long


def test_dataset_length():
    assert len(ShapeDataset(num_samples=7, cache=False)) == 7
    assert len(SceneDataset(num_samples=5, cache=False)) == 5
