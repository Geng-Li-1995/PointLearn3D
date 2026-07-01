"""3D primitive geometry, point cloud utilities, and scene/shape generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config.config import NUM_SHAPE_CLASSES

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _sphere_surface_single(n_points=1024, radius=1.0):
    phi = np.arccos(1 - 2 * np.random.rand(n_points))
    theta = 2 * np.pi * np.random.rand(n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def generate_sphere_surface(n_samples, n_points=1024, radius=1.0):
    samples = [_sphere_surface_single(n_points, radius) for _ in range(n_samples)]
    return np.stack(samples, axis=0)


def _cuboid_surface_single(n_points=1024, size=(1.0, 1.0, 1.0)):
    lx, ly, lz = size
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    points_per_face = n_points // 6
    faces = []
    for sign in [-1, 1]:
        y = np.random.uniform(-hy, hy, (points_per_face, 1))
        z = np.random.uniform(-hz, hz, (points_per_face, 1))
        x = np.full((points_per_face, 1), sign * hx)
        faces.append(np.hstack([x, y, z]))
    for sign in [-1, 1]:
        x = np.random.uniform(-hx, hx, (points_per_face, 1))
        z = np.random.uniform(-hz, hz, (points_per_face, 1))
        y = np.full((points_per_face, 1), sign * hy)
        faces.append(np.hstack([x, y, z]))
    for sign in [-1, 1]:
        x = np.random.uniform(-hx, hx, (points_per_face, 1))
        y = np.random.uniform(-hy, hy, (points_per_face, 1))
        z = np.full((points_per_face, 1), sign * hz)
        faces.append(np.hstack([x, y, z]))
    points = np.vstack(faces)
    if points.shape[0] < n_points:
        idx = np.random.choice(points.shape[0], n_points - points.shape[0])
        points = np.vstack([points, points[idx]])
    else:
        points = points[:n_points]
    return points.astype(np.float32)


def generate_cuboid_surface(n_samples, n_points=1024, size=(1.0, 1.0, 1.0)):
    samples = [_cuboid_surface_single(n_points, size) for _ in range(n_samples)]
    return np.stack(samples, axis=0)


def _cylinder_cap(radius, cap_points, z_val):
    r = np.sqrt(np.random.rand(cap_points)) * radius
    theta = np.random.uniform(0, 2 * np.pi, cap_points)
    return np.stack((r * np.cos(theta), r * np.sin(theta), np.full(cap_points, z_val)), axis=1)


def _cylinder_surface_single(n_points=1024, radius=0.5, height=1.0):
    side_points = n_points * 4 // 6
    cap_points = (n_points - side_points) // 2
    theta = np.random.uniform(0, 2 * np.pi, side_points)
    z = np.random.uniform(-height / 2, height / 2, side_points)
    side = np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=1)
    return np.vstack(
        (side, _cylinder_cap(radius, cap_points, height / 2), _cylinder_cap(radius, cap_points, -height / 2))
    ).astype(np.float32)


def generate_cylinder_surface(n_samples, n_points=1024, radius=0.5, height=1.0):
    samples = [_cylinder_surface_single(n_points, radius, height) for _ in range(n_samples)]
    return np.stack(samples, axis=0)


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def apply_transform(points, R=None, t=None):
    if R is None:
        R = np.eye(3, dtype=np.float32)
    if t is None:
        t = np.zeros(3, dtype=np.float32)
    return points @ R.T + t


def random_rotation(max_angle=np.pi):
    rx = np.random.uniform(-max_angle, max_angle)
    ry = np.random.uniform(-max_angle, max_angle)
    rz = np.random.uniform(0, 2 * np.pi)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def random_translation(low=-10.0, high=10.0):
    return np.random.uniform(low, high, size=3).astype(np.float32)


def transform_point_cloud(points, translation, yaw=None, tilt=None, yaw_range=(0.0, 2 * np.pi), tilt_range=0.1):
    if yaw is None:
        yaw = np.random.uniform(*yaw_range)
    if tilt is None:
        tilt = np.random.uniform(-tilt_range, tilt_range)
    return apply_transform(points, rot_z(yaw) @ rot_x(tilt), translation)


# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------


def resample_indices(n: int, num_points: int) -> np.ndarray:
    if n == num_points:
        return np.arange(n)
    return np.random.choice(n, num_points, replace=n < num_points)


def resample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    return points[resample_indices(len(points), num_points)]


def resample_point_cloud(points, labels, num_points):
    idx = resample_indices(len(points), num_points)
    return points[idx], labels[idx]


def normalize_numpy(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / (scale + 1e-6)


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    squeeze = points.dim() == 2
    if squeeze:
        points = points.unsqueeze(0)
    centroid = points.mean(dim=1, keepdim=True)
    points = points - centroid
    scale = torch.norm(points, dim=-1).mean(dim=1, keepdim=True)
    points = points / (scale.unsqueeze(-1) + 1e-6)
    return points.squeeze(0) if squeeze else points


def scene_objects_to_arrays(objects):
    scene_points, scene_labels = [], []
    for obj in objects:
        scene_points.append(obj.points)
        scene_labels.append(np.full(len(obj.points), obj.label, dtype=np.int64))
    scene = np.concatenate(scene_points, axis=0)
    labels = np.concatenate(scene_labels, axis=0)
    perm = np.random.permutation(len(scene))
    return scene[perm].astype(np.float32), labels[perm]


# ---------------------------------------------------------------------------
# Scene / shape generators
# ---------------------------------------------------------------------------


@dataclass
class SceneObject:
    """One placed object in a generated scene."""

    points: np.ndarray
    label: int
    obj_type: str = "unknown"
    center: np.ndarray | None = None
    footprint_radius: float = 0.0


class VoxelEngine:
    """Coarse occupancy and footprint checks for non-overlapping placement."""

    def __init__(self, voxel_size: float = 0.05, clearance: float = 0.2):
        self.voxel_size = voxel_size
        self.clearance = clearance
        self.occupied: set[tuple[int, ...]] = set()
        self.footprints: list[tuple[np.ndarray, float]] = []

    def _unique_keys(self, points: np.ndarray) -> set[tuple[int, ...]]:
        voxels = np.floor(points / self.voxel_size).astype(np.int32)
        return {tuple(v) for v in np.unique(voxels, axis=0)}

    def _query_footprint_collision(self, center: np.ndarray, radius: float) -> bool:
        center_xy = center[:2]
        for placed_center_xy, placed_radius in self.footprints:
            min_distance = radius + placed_radius + self.clearance
            if np.linalg.norm(center_xy - placed_center_xy) < min_distance:
                return True
        return False

    def query_collision(
        self,
        points: np.ndarray,
        center: np.ndarray | None = None,
        footprint_radius: float | None = None,
    ) -> bool:
        if center is not None and footprint_radius is not None:
            if self._query_footprint_collision(center, footprint_radius):
                return True
        return bool(self._unique_keys(points) & self.occupied)

    def add_object(
        self,
        points: np.ndarray,
        center: np.ndarray | None = None,
        footprint_radius: float | None = None,
    ) -> None:
        self.occupied.update(self._unique_keys(points))
        if center is not None and footprint_radius is not None:
            self.footprints.append((center[:2].astype(np.float32), float(footprint_radius)))

    def reset(self) -> None:
        self.occupied.clear()
        self.footprints.clear()


@dataclass
class SceneGenerationConfig:
    """Default counts and sampling density for scene synthesis."""

    total_objects: int = 12
    min_cubes: int = 3
    max_cubes: int = 5
    points_per_object: int = 600


class SceneGenerator:
    """Place random cuboids, cylinders, and spheres without voxel collisions."""

    def __init__(
        self,
        max_attempts: int = 200,
        voxel_size: float = 0.05,
        clearance: float = 0.2,
        bounds: tuple = ((-10, -10, -0.1), (10, 10, 0.1)),
        scene_config: SceneGenerationConfig | None = None,
    ):
        self.max_attempts = max_attempts
        self.bounds = bounds
        self.scene_config = scene_config or SceneGenerationConfig()
        self.engine = VoxelEngine(voxel_size=voxel_size, clearance=clearance)
        self.objects: list[SceneObject] = []

    def _random_pose(self) -> tuple[np.ndarray, float, float]:
        low, high = self.bounds
        translation = np.random.uniform(low, high).astype(np.float32)
        return translation, np.random.uniform(0, 2 * np.pi), np.random.uniform(-0.1, 0.1)

    def _place_object(self, points: np.ndarray, label: int, obj_type: str = "unknown") -> bool:
        for _ in range(self.max_attempts):
            translation, yaw, tilt = self._random_pose()
            transformed = transform_point_cloud(points, translation=translation, yaw=yaw, tilt=tilt)
            footprint_radius = float(np.max(np.linalg.norm(transformed[:, :2] - translation[:2], axis=1)))
            if not self.engine.query_collision(transformed, translation, footprint_radius):
                self.engine.add_object(transformed, translation, footprint_radius)
                self.objects.append(SceneObject(transformed, label, obj_type, translation, footprint_radius))
                return True
        return False

    def reset(self) -> None:
        self.engine.reset()
        self.objects = []

    def generate_scene(self) -> list[SceneObject]:
        self.reset()
        cfg = self.scene_config
        n_cubes = np.random.randint(cfg.min_cubes, cfg.max_cubes + 1)
        n_cyl = np.random.randint(cfg.min_cubes, cfg.max_cubes + 1)
        n_sph = cfg.total_objects - n_cubes - n_cyl

        for _ in range(n_cubes):
            size = np.random.uniform(1.5, 2.5, size=3)
            pts = generate_cuboid_surface(1, cfg.points_per_object, size=size)[0]
            self._place_object(pts, label=0, obj_type="cuboid")

        for _ in range(n_cyl):
            pts = generate_cylinder_surface(
                1, cfg.points_per_object,
                radius=np.random.uniform(0.25, 0.4),
                height=np.random.uniform(1.5, 2.0),
            )[0]
            self._place_object(pts, label=1, obj_type="cylinder")

        for _ in range(n_sph):
            pts = generate_sphere_surface(
                1, cfg.points_per_object, radius=np.random.uniform(0.3, 1.0),
            )[0]
            self._place_object(pts, label=2, obj_type="sphere")

        return self.objects

    def generate_scene_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return scene_objects_to_arrays(self.generate_scene())


class ShapeGenerator:
    """Sample one primitive, apply random rigid transform, return points + class id."""

    def __init__(
        self,
        n_points: int = 1024,
        translation_range: tuple[float, float] = (-5.0, 5.0),
        max_rotation: float = np.pi,
    ):
        self.n_points = n_points
        self.translation_range = translation_range
        self.max_rotation = max_rotation

    def _sample_primitive(self, shape_class: int) -> np.ndarray:
        if shape_class == 0:
            size = np.random.uniform(1.0, 2.5, size=3)
            return generate_cuboid_surface(1, self.n_points, size=size)[0]
        if shape_class == 1:
            return generate_cylinder_surface(
                1, self.n_points,
                radius=np.random.uniform(0.3, 0.8),
                height=np.random.uniform(1.0, 2.0),
            )[0]
        return generate_sphere_surface(1, self.n_points, radius=np.random.uniform(0.5, 1.2))[0]

    def generate(self, shape_class: int | None = None) -> tuple[np.ndarray, int]:
        if shape_class is None:
            shape_class = int(np.random.randint(0, NUM_SHAPE_CLASSES))
        points = self._sample_primitive(shape_class)
        points = apply_transform(
            points,
            random_rotation(max_angle=self.max_rotation),
            random_translation(*self.translation_range),
        )
        return points.astype(np.float32), shape_class
