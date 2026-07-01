"""Project paths, constants, and pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from config.input import Input

SHAPE_CLASS_NAMES = {0: "cuboid", 1: "cylinder", 2: "sphere"}
NUM_SHAPE_CLASSES = 3
CLASS_COLORS = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
}
DEFAULT_COLOR = (0.7, 0.7, 0.7)

DATA_DIR = Path("data")
SHAPE_DATA_DIR = DATA_DIR / "shape"
SCENE_DATA_DIR = DATA_DIR / "scene"
RESULT_DIR = Path("result")
RESULT_SHAPE_DIR = RESULT_DIR / "shape"
RESULT_SCENE_DIR = RESULT_DIR / "scene"
MODELS_DIR = RESULT_DIR / "models"
TRAINING_PLOTS_DIR = RESULT_DIR / "training"

TRAINING_STAGE_LABELS = {
    "shape": "Shape Classification",
    "scene": "Scene Segmentation",
    "single_shape": "Shape Classification",
    "scene_segmenter": "Scene Segmentation",
}


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_num_workers(num_workers: int) -> int:
    if num_workers <= 0:
        cpus = os.cpu_count() or 1
        return 2 if cpus >= 4 else 1
    return num_workers


def resolve_preload_workers(preload_workers: int) -> int:
    if preload_workers <= 0:
        return os.cpu_count() or 1
    return preload_workers


def configure_cpu_threads(num_workers: int = 0, cpu_threads: int = 0) -> int:
    cpus = os.cpu_count() or 1
    if cpu_threads > 0:
        threads = min(cpu_threads, cpus)
    else:
        threads = max(1, cpus - max(0, num_workers))
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, threads // 2))
        except RuntimeError:
            pass
    return threads


def _validate(inp: Input) -> None:
    will_prepare = inp.prepare_data and (inp.prepare_shape or inp.prepare_scene)
    will_train = inp.train and (inp.train_shape or inp.train_scene)
    will_preview = inp.preview_shape or inp.preview_scene
    if not (
        will_prepare
        or will_train
        or will_preview
        or inp.export_examples
        or inp.plot_training_curves
        or inp.recognize_scene
    ):
        raise ValueError(
            "Enable at least one action in config/input.py: "
            "prepare_data, train, preview_*, export_examples, plot_training_curves, or recognize_scene."
        )
    if inp.prepare_data and not (inp.prepare_shape or inp.prepare_scene):
        raise ValueError("prepare_data=True requires prepare_shape or prepare_scene.")
    if inp.train and not (inp.train_shape or inp.train_scene):
        raise ValueError("train=True requires train_shape or train_scene.")
    if inp.early_stop_min_epochs < 1:
        raise ValueError("early_stop_min_epochs must be >= 1.")
    if inp.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1.")
    if inp.early_stop_min_delta < 0 or inp.early_stop_loss_min_delta < 0:
        raise ValueError("early stop deltas must be >= 0.")
    if inp.scene_k_neighbors < 1:
        raise ValueError("scene_k_neighbors must be >= 1.")
    if inp.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if inp.cpu_threads < 0:
        raise ValueError("cpu_threads must be >= 0.")


def run(inp: Input | None = None) -> None:
    """Preview/export -> dataset preparation -> training -> curve plots."""
    from learning.datasets import SceneDataset, ShapeDataset
    from learning.train import train_scene, train_shape
    from learning.visualize import (
        export_visualization_examples,
        plot_training_curves,
        preview_scenes,
        preview_shapes,
    )
    from learning.recognition import export_recognition_examples

    inp = inp or Input()
    _validate(inp)
    preload_workers = resolve_preload_workers(inp.preload_workers)
    print(
        f"regen={inp.regen} | prepare_data={inp.prepare_data} | train={inp.train} | "
        f"shape={inp.prepare_shape or inp.train_shape} | "
        f"scene={inp.prepare_scene or inp.train_scene} | preload_workers={preload_workers}"
    )

    if inp.preview_shape:
        preview_shapes(n_points=inp.num_points_shape, seed=inp.seed)

    if inp.preview_scene:
        preview_scenes(count=inp.scene_preview_count, seed=inp.seed)

    if inp.export_examples:
        print("=== Exporting visualization examples ===")
        export_visualization_examples(scene_count=inp.scene_export_count, seed=inp.seed or 42)
        print("Export complete.")

    shape_prepared = scene_prepared = False

    if inp.prepare_data:
        if inp.prepare_shape:
            print("=== Preparing shape data ===")
            ShapeDataset(
                num_samples=inp.num_samples_shape,
                num_points=inp.num_points_shape,
                seed=inp.seed,
                regen=inp.regen,
                cache=inp.cache,
                preload_workers=preload_workers,
            )
            shape_prepared = True
        if inp.prepare_scene:
            print("=== Preparing scene data ===")
            SceneDataset(
                num_samples=inp.num_samples_scene,
                num_points=inp.num_points_scene,
                seed=inp.seed,
                regen=inp.regen,
                cache=inp.cache,
                preload_workers=preload_workers,
            )
            scene_prepared = True

    if inp.train:
        if inp.train_shape:
            print("=== Training shape classifier ===")
            train_shape(inp, regen=False if shape_prepared else None)
        if inp.train_scene:
            print("=== Training scene segmenter ===")
            train_scene(inp, regen=False if scene_prepared else None)

    if inp.plot_training_curves:
        print("=== Plotting training curves ===")
        plot_training_curves()

    if inp.recognize_scene:
        print("=== Layered scene recognition ===")
        export_recognition_examples(
            count=inp.recognition_scene_count,
            seed=inp.seed or 42,
            device=inp.device,
            cluster_eps=inp.recognition_cluster_eps,
            min_cluster_points=inp.recognition_min_cluster_points,
            refine_geometry=inp.recognition_refine_geometry,
            num_points=inp.num_points_scene,
        )
