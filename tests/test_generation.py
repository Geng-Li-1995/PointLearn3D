"""Tests for geometry and point cloud generation."""

import numpy as np

from config.config import NUM_SHAPE_CLASSES
from simulation.generation import (
    SceneGenerator,
    ShapeGenerator,
    normalize_numpy,
    resample_point_cloud,
    resample_points,
    scene_objects_to_arrays,
)


def test_shape_generator_output_shape_and_label():
    gen = ShapeGenerator(n_points=256)
    for label in range(NUM_SHAPE_CLASSES):
        points, out_label = gen.generate(shape_class=label)
        assert points.shape == (256, 3)
        assert points.dtype == np.float32
        assert out_label == label


def test_shape_generator_random_label():
    points, label = ShapeGenerator(n_points=64).generate()
    assert points.shape == (64, 3)
    assert 0 <= label < NUM_SHAPE_CLASSES


def test_scene_generator_places_objects():
    objects = SceneGenerator().generate_scene()
    assert len(objects) > 0
    for obj in objects:
        assert obj.points.ndim == 2 and obj.points.shape[1] == 3
        assert 0 <= obj.label < NUM_SHAPE_CLASSES


def test_scene_objects_to_arrays():
    objects = SceneGenerator().generate_scene()
    points, labels = scene_objects_to_arrays(objects)
    assert points.shape[0] == labels.shape[0]
    assert points.dtype == np.float32
    assert labels.dtype == np.int64


def test_resample_points_fixed_count():
    pts = np.random.randn(500, 3).astype(np.float32)
    assert resample_points(pts, 128).shape == (128, 3)


def test_resample_point_cloud_preserves_label_alignment():
    points = np.random.randn(200, 3).astype(np.float32)
    labels = np.random.randint(0, 3, size=200)
    new_pts, new_labels = resample_point_cloud(points, labels, 64)
    assert new_pts.shape == (64, 3)
    assert new_labels.shape == (64,)


def test_normalize_numpy_unit_scale():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    normed = normalize_numpy(pts)
    assert np.isfinite(normed).all()
    assert np.max(np.linalg.norm(normed, axis=1)) <= 1.0 + 1e-5
