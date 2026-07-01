"""Tests for config helpers and input validation."""

import pytest
import torch

from config.config import _validate, configure_cpu_threads, resolve_device, resolve_num_workers, resolve_preload_workers
from config.input import Input


def test_validate_rejects_empty_pipeline():
    inp = Input(
        prepare_data=False,
        train=False,
        preview_shape=False,
        preview_scene=False,
        export_examples=False,
        plot_training_curves=False,
        recognize_scene=False,
    )
    with pytest.raises(ValueError, match="Enable at least one action"):
        _validate(inp)


def test_validate_prepare_data_requires_target():
    inp = Input(prepare_data=True, prepare_shape=False, prepare_scene=False)
    with pytest.raises(ValueError, match="prepare_shape or prepare_scene"):
        _validate(inp)


def test_validate_train_requires_target():
    inp = Input(train=True, train_shape=False, train_scene=False)
    with pytest.raises(ValueError, match="train_shape or train_scene"):
        _validate(inp)


def test_validate_rejects_invalid_early_stop_settings():
    with pytest.raises(ValueError, match="early_stop_min_epochs"):
        _validate(Input(early_stop_min_epochs=0))
    with pytest.raises(ValueError, match="early_stop_patience"):
        _validate(Input(early_stop_patience=0))
    with pytest.raises(ValueError, match="early stop deltas"):
        _validate(Input(early_stop_min_delta=-0.1))


def test_validate_rejects_invalid_scene_neighbor_count():
    with pytest.raises(ValueError, match="scene_k_neighbors"):
        _validate(Input(scene_k_neighbors=0))


def test_validate_rejects_invalid_worker_settings():
    with pytest.raises(ValueError, match="num_workers"):
        _validate(Input(num_workers=-1))
    with pytest.raises(ValueError, match="cpu_threads"):
        _validate(Input(cpu_threads=-1))


def test_validate_accepts_preview_only():
    _validate(Input(
        prepare_data=False,
        train=False,
        preview_shape=True,
        preview_scene=False,
        export_examples=False,
        plot_training_curves=False,
    ))


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_uses_mps_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_resolve_num_workers_auto():
    assert resolve_num_workers(0) >= 1


def test_configure_cpu_threads_explicit():
    assert configure_cpu_threads(num_workers=2, cpu_threads=8) >= 1
    assert configure_cpu_threads(num_workers=2, cpu_threads=8) <= 8


def test_configure_cpu_threads_auto_leaves_loader_cores():
    threads_with_workers = configure_cpu_threads(num_workers=2, cpu_threads=0)
    threads_without_workers = configure_cpu_threads(num_workers=0, cpu_threads=0)
    assert threads_with_workers <= threads_without_workers


def test_resolve_preload_workers_auto():
    assert resolve_preload_workers(0) >= 1
