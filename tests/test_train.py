"""Tests for model forward passes and training helpers."""

import torch

from config.config import NUM_SHAPE_CLASSES
from config.input import Input
from learning.train import SceneSegmenter, ShapeClassifier, _should_stop_training


def test_shape_classifier_output_shape():
    x = torch.randn(4, 512, 3)
    out = ShapeClassifier(num_classes=NUM_SHAPE_CLASSES)(x)
    assert out.shape == (4, NUM_SHAPE_CLASSES)


def test_scene_segmenter_per_point_output():
    batch, num_points = 2, 1024
    x = torch.randn(batch, num_points, 3)
    out = SceneSegmenter(num_classes=NUM_SHAPE_CLASSES)(x)
    assert out.shape == (batch, num_points, NUM_SHAPE_CLASSES)


def test_early_stop_on_target_accuracy():
    inp = Input(target_accuracy=0.95, early_stop=True)
    stop, best, patience = _should_stop_training(inp, acc=0.96, best_acc=0.5, patience=0)
    assert stop is True


def test_early_stop_patience_resets_on_improvement():
    inp = Input(early_stop=True, early_stop_patience=3, early_stop_min_delta=0.01)
    stop, best, patience = _should_stop_training(inp, acc=0.80, best_acc=0.70, patience=2)
    assert stop is False
    assert patience == 0
    assert best == 0.80


def test_no_early_stop_when_disabled():
    inp = Input(early_stop=False)
    stop, best, patience = _should_stop_training(inp, acc=0.5, best_acc=0.4, patience=10)
    assert stop is False
    assert best == 0.5
