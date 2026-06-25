"""PyTorch datasets with optional disk cache and parallel preloading."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import SCENE_DATA_DIR, SHAPE_DATA_DIR
from simulation.generation import (
    SceneGenerator,
    ShapeGenerator,
    normalize_numpy,
    resample_point_cloud,
    resample_points,
)

T = TypeVar("T")
R = TypeVar("R")


def _limit_blas_threads() -> None:
    """Avoid oversubscription when many worker processes run NumPy/BLAS."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _parallel_map(
    worker: Callable[[T], R], items: Iterable[T], desc: str, max_workers: int | None = None,
) -> list[R]:
    items = list(items)
    if not items:
        return []
    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    chunksize = max(1, len(items) // (workers * 4))
    print(f"{desc}: {len(items)} samples with {workers} processes...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, items, chunksize=chunksize))


def _shape_sample_task(args: tuple[int, int, int | None]) -> tuple[int, object, int]:
    _limit_blas_threads()
    idx, num_points, seed = args
    if seed is not None:
        np.random.seed(seed + idx)
    points, label = ShapeGenerator(n_points=num_points).generate()
    return idx, resample_points(points, num_points), label


def _scene_sample_task(args: tuple[int, int, int | None]) -> tuple[int, object, object]:
    _limit_blas_threads()
    idx, num_points, seed = args
    if seed is not None:
        np.random.seed(seed + idx)
    scene, labels = SceneGenerator().generate_scene_arrays()
    return idx, *resample_point_cloud(scene, labels, num_points)


def _preload_shapes(
    num_samples: int, num_points: int, seed: int | None, preload_workers: int | None = None,
) -> dict:
    cache = {}
    tasks = [(i, num_points, seed) for i in range(num_samples)]
    for idx, points, label in _parallel_map(
        _shape_sample_task, tasks, "Caching shapes", max_workers=preload_workers,
    ):
        cache[idx] = (points, label)
    print(f"Shape cache ready ({len(cache)} samples).")
    return cache


def _preload_scene(
    num_samples: int, num_points: int, seed: int | None, preload_workers: int | None = None,
) -> dict:
    cache = {}
    tasks = [(i, num_points, seed) for i in range(num_samples)]
    for idx, scene, labels in _parallel_map(
        _scene_sample_task, tasks, "Caching scene", max_workers=preload_workers,
    ):
        cache[idx] = (scene, labels)
    print(f"Scene cache ready ({len(cache)} samples).")
    return cache


def _save_npz(path, points, labels, num_points: int, seed: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points.astype(np.float32),
        labels=labels,
        num_points=np.int32(num_points),
        seed=np.int32(-1 if seed is None else seed),
        count=np.int32(len(labels) if labels.ndim == 1 else len(points)),
    )


def _load_npz(path) -> tuple[dict, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"No saved dataset at {path}")
    data = np.load(path)
    count, num_points = int(data["count"]), int(data["num_points"])
    points, labels = data["points"], data["labels"]
    if labels.ndim == 1:
        cache = {i: (points[i], int(labels[i])) for i in range(count)}
    else:
        cache = {i: (points[i], labels[i]) for i in range(count)}
    return cache, num_points, count


def _load_or_create(path, count, num_points, seed, regen, preload_fn, label, preload_workers=None) -> dict:
    if path.exists() and not regen:
        cache, saved_points, saved_count = _load_npz(path)
        if saved_count == count and saved_points == num_points:
            print(f"Loaded {saved_count} {label} from {path}")
            return cache
        print(
            f"Cache mismatch at {path} "
            f"(saved: count={saved_count}, num_points={saved_points}; "
            f"requested: count={count}, num_points={num_points}). Rebuilding..."
        )
    elif regen and path.exists():
        print(f"regen=True: rebuilding {label} at {path}")

    cache = preload_fn(count, num_points, seed, preload_workers)
    indices = sorted(cache.keys())
    points = np.stack([cache[i][0] for i in indices])
    labels = (
        np.array([cache[i][1] for i in indices], dtype=np.int64)
        if label == "shape"
        else np.stack([cache[i][1] for i in indices]).astype(np.int64)
    )
    _save_npz(path, points, labels, num_points, seed)
    print(f"Saved {len(cache)} {label} to {path}")
    return cache


class ShapeDataset(Dataset):
    """Single transformed primitives with global shape labels."""

    def __init__(
        self,
        num_samples: int = 3000,
        num_points: int = 1024,
        normalize: bool = False,
        cache: bool = True,
        seed: int | None = None,
        regen: bool = False,
        preload_workers: int | None = None,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.normalize = normalize
        self.seed = seed
        self.on_the_fly = not cache
        self.generator = None if cache else ShapeGenerator(n_points=num_points)
        self._cache = None
        if cache:
            self._cache = _load_or_create(
                SHAPE_DATA_DIR / "dataset.npz",
                num_samples, num_points, seed, regen, _preload_shapes, "shape", preload_workers,
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self._cache is not None:
            points, label = self._cache[idx]
        else:
            if self.seed is not None:
                np.random.seed(self.seed + idx)
            points, label = self.generator.generate()
            points = resample_points(points, self.num_points)
        if self.normalize:
            points = normalize_numpy(points)
        return torch.from_numpy(points.copy()).float(), torch.tensor(label, dtype=torch.long)


class SceneDataset(Dataset):
    """Multi-object scenes with per-point segmentation labels."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 4096,
        normalize: bool = False,
        cache: bool = True,
        seed: int | None = None,
        regen: bool = False,
        preload_workers: int | None = None,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.normalize = normalize
        self.seed = seed
        self.on_the_fly = not cache
        self.generator = None if cache else SceneGenerator()
        self._cache = None
        if cache:
            self._cache = _load_or_create(
                SCENE_DATA_DIR / "dataset.npz",
                num_samples, num_points, seed, regen, _preload_scene, "scene", preload_workers,
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self._cache is not None:
            scene, labels = self._cache[idx]
        else:
            if self.seed is not None:
                np.random.seed(self.seed + idx)
            scene, labels = self.generator.generate_scene_arrays()
            scene, labels = resample_point_cloud(scene, labels, self.num_points)
        if self.normalize:
            scene = normalize_numpy(scene)
        return torch.from_numpy(scene.copy()).float(), torch.from_numpy(labels.copy()).long()
