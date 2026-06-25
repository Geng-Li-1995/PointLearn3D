"""Tests for config helpers and input validation."""

import pytest

from config.config import _validate, resolve_device, resolve_num_workers, resolve_preload_workers
from config.input import Input


def test_validate_rejects_empty_pipeline():
    inp = Input(
        prepare_data=False,
        train=False,
        preview_shape=False,
        preview_scene=False,
        export_examples=False,
        plot_training_curves=False,
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


def test_resolve_num_workers_auto():
    assert resolve_num_workers(0) >= 1


def test_resolve_preload_workers_auto():
    assert resolve_preload_workers(0) >= 1
