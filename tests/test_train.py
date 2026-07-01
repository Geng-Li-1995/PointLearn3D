"""Tests for model forward passes and training helpers."""

import torch

from config.config import NUM_SHAPE_CLASSES
from config.input import Input
from learning.models import SceneSegmenter, ShapeClassifier
from learning.train import EarlyStopState, _scene_class_weights, _should_stop_training


def test_shape_classifier_output_shape():
    x = torch.randn(4, 512, 3)
    out = ShapeClassifier(num_classes=NUM_SHAPE_CLASSES)(x)
    assert out.shape == (4, NUM_SHAPE_CLASSES)


def test_scene_segmenter_per_point_output():
    batch, num_points = 2, 1024
    x = torch.randn(batch, num_points, 3)
    out = SceneSegmenter(num_classes=NUM_SHAPE_CLASSES)(x)
    assert out.shape == (batch, num_points, NUM_SHAPE_CLASSES)


def test_scene_segmenter_accepts_custom_neighbor_count():
    model = SceneSegmenter(num_classes=NUM_SHAPE_CLASSES, k_neighbors=8)
    assert model.backbone.k_neighbors == 8


def test_scene_segmenter_uses_point_features():
    x = torch.zeros(1, 8, 3)
    x[0, :, 0] = torch.linspace(-1.0, 1.0, 8)
    out = SceneSegmenter(num_classes=NUM_SHAPE_CLASSES)(x)
    assert not torch.allclose(out[:, :1, :].expand_as(out), out)


def test_scene_class_weights_balance_cached_labels():
    class Dataset:
        _cache = {
            0: (None, torch.tensor([0, 1, 2, 2])),
            1: (None, torch.tensor([0, 2, 2, 2])),
        }

    weights = _scene_class_weights(Dataset(), "cpu")
    assert weights[0] > weights[2]
    assert weights[1] > weights[0]


def test_early_stop_on_target_accuracy():
    inp = Input(target_accuracy=0.95, early_stop=True, early_stop_min_epochs=1)
    stop, state, improved = _should_stop_training(inp, loss=0.1, acc=0.96, epoch=0, state=EarlyStopState())
    assert stop is True
    assert improved is True
    assert state.best_acc == 0.96


def test_early_stop_patience_resets_on_improvement():
    inp = Input(early_stop=True, early_stop_patience=3, early_stop_min_delta=0.01)
    state = EarlyStopState(best_acc=0.70, best_loss=1.0, best_epoch=0, patience=2)
    stop, state, improved = _should_stop_training(inp, loss=0.9, acc=0.80, epoch=1, state=state)
    assert stop is False
    assert improved is True
    assert state.patience == 0
    assert state.best_acc == 0.80


def test_early_stop_uses_loss_when_accuracy_is_flat():
    inp = Input(early_stop=True, early_stop_patience=3, early_stop_min_delta=0.01, early_stop_loss_min_delta=0.01)
    state = EarlyStopState(best_acc=0.80, best_loss=1.0, best_epoch=0, patience=2)
    stop, state, improved = _should_stop_training(inp, loss=0.8, acc=0.805, epoch=1, state=state)
    assert stop is False
    assert improved is True
    assert state.patience == 0
    assert state.best_loss == 0.8


def test_early_stop_waits_for_min_epochs():
    inp = Input(early_stop=True, early_stop_patience=1, early_stop_min_epochs=3)
    state = EarlyStopState(best_acc=0.80, best_loss=1.0, best_epoch=0, patience=0)
    stop, state, improved = _should_stop_training(inp, loss=1.1, acc=0.70, epoch=1, state=state)
    assert stop is False
    assert improved is False


def test_no_early_stop_when_disabled():
    inp = Input(early_stop=False)
    state = EarlyStopState(best_acc=0.4, best_loss=1.0, best_epoch=0, patience=10)
    stop, state, improved = _should_stop_training(inp, loss=1.1, acc=0.5, epoch=10, state=state)
    assert stop is False
    assert state.best_acc == 0.5
