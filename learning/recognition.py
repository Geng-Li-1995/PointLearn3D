"""Layered scene recognition: instances -> shape classification -> geometry fits."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pointlearn3d-matplotlib")

import matplotlib
import numpy as np
import torch

from config.config import CLASS_COLORS, DEFAULT_COLOR, MODELS_DIR, NUM_SHAPE_CLASSES, RESULT_DIR, SHAPE_CLASS_NAMES, resolve_device
from learning.models import ShapeClassifier
from simulation.generation import SceneGenerator, normalize_point_cloud, resample_point_cloud

matplotlib.use("Agg")

import matplotlib.pyplot as plt

INSTANCE_COLORS = [
    (0.90, 0.10, 0.12),
    (0.12, 0.47, 0.71),
    (0.17, 0.63, 0.17),
    (1.00, 0.50, 0.05),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
]
POINT_SIZE = 2.0
PLOT_DPI = 150


@dataclass
class GeometryFit:
    label: int
    name: str
    confidence: float
    residual: float
    parameters: dict[str, list[float] | float]


@dataclass
class RecognizedInstance:
    instance_id: int
    semantic_label: int
    semantic_name: str
    geometry_label: int
    geometry_name: str
    confidence: float
    num_points: int
    centroid: list[float]
    geometry: GeometryFit


def _set_scene_axes(ax, points: np.ndarray, pad: float = 0.06) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    margin = spans * pad
    for i, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(mins[i] - margin[i], maxs[i] + margin[i])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(tuple(spans))


def save_labeled_scene_plot(
    points: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    *,
    title: str,
    use_instance_palette: bool = False,
    dpi: int = PLOT_DPI,
) -> Path:
    """Save a point cloud colored by class labels or instance ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = (
        {i: INSTANCE_COLORS[i % len(INSTANCE_COLORS)] for i in np.unique(labels) if i >= 0}
        if use_instance_palette
        else CLASS_COLORS
    )
    colors = np.array([palette.get(int(label), DEFAULT_COLOR) for label in labels], dtype=np.float32)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=POINT_SIZE, linewidths=0, alpha=0.9)
    _set_scene_axes(ax, points)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_shape_classifier(path: str | Path = MODELS_DIR / "shape.pt", device: str = "auto") -> ShapeClassifier:
    """Load the trained single-object shape classifier."""
    resolved_device = resolve_device(device)
    model = ShapeClassifier(num_classes=NUM_SHAPE_CLASSES).to(resolved_device)
    state = torch.load(path, map_location=resolved_device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def classify_instance(model: ShapeClassifier, points: np.ndarray, device: str = "auto") -> tuple[int, float]:
    """Classify one object instance and return class id plus softmax confidence."""
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(resolved_device)
    probs = torch.softmax(model(normalize_point_cloud(tensor)), dim=-1).squeeze(0)
    confidence, label = torch.max(probs, dim=-1)
    return int(label.cpu()), float(confidence.cpu())


def dbscan_points(points: np.ndarray, eps: float = 0.65, min_points: int = 20) -> np.ndarray:
    """Small NumPy DBSCAN for point-cloud instances."""
    labels = np.full(len(points), -1, dtype=np.int64)
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    if len(points) == 0:
        return labels

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    neighbors = [np.flatnonzero(distances[i] <= eps) for i in range(len(points))]

    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_points:
            continue

        labels[i] = cluster_id
        seeds = list(neighbors[i])
        seed_set = set(int(n) for n in seeds)
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_points:
                    for n in neighbors[j]:
                        n = int(n)
                        if n not in seed_set:
                            seeds.append(n)
                            seed_set.add(n)
            if labels[j] == -1:
                labels[j] = cluster_id
            cursor += 1
        cluster_id += 1
    return labels


def cluster_spatial_instances(points: np.ndarray, eps: float = 0.65, min_points: int = 20) -> list[np.ndarray]:
    """Cluster the full scene spatially, ignoring semantic labels."""
    cluster_labels = dbscan_points(points, eps=eps, min_points=min_points)
    return [np.flatnonzero(cluster_labels == cluster_id) for cluster_id in sorted(set(cluster_labels) - {-1})]


def _pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, values, vt = np.linalg.svd(centered, full_matrices=False)
    return center, values, vt


def _sphere_fit(points: np.ndarray) -> GeometryFit:
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = float(np.mean(radii))
    residual = float(np.std(radii) / (radius + 1e-6))
    return GeometryFit(
        label=2,
        name=SHAPE_CLASS_NAMES[2],
        confidence=float(np.exp(-3.0 * residual)),
        residual=residual,
        parameters={"center": center.tolist(), "radius": radius},
    )


def _cylinder_fit(points: np.ndarray) -> GeometryFit:
    center, _, axes = _pca_axes(points)
    axis = axes[0]
    centered = points - center
    height_coords = centered @ axis
    radial = centered - np.outer(height_coords, axis)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.mean(radii))
    height = float(np.ptp(height_coords))
    residual = float(np.std(radii) / (radius + 1e-6))
    return GeometryFit(
        label=1,
        name=SHAPE_CLASS_NAMES[1],
        confidence=float(np.exp(-3.0 * residual)),
        residual=residual,
        parameters={"center": center.tolist(), "axis": axis.tolist(), "radius": radius, "height": height},
    )


def _cuboid_fit(points: np.ndarray) -> GeometryFit:
    center, _, axes = _pca_axes(points)
    local = (points - center) @ axes.T
    half_extents = np.max(np.abs(local), axis=0)
    face_distance = np.min(np.abs(np.abs(local) - half_extents), axis=1)
    scale = float(np.mean(half_extents) + 1e-6)
    residual = float(np.mean(face_distance) / scale)
    return GeometryFit(
        label=0,
        name=SHAPE_CLASS_NAMES[0],
        confidence=float(np.exp(-3.0 * residual)),
        residual=residual,
        parameters={"center": center.tolist(), "axes": axes.tolist(), "half_extents": half_extents.tolist()},
    )


def fit_geometry(points: np.ndarray, semantic_label: int | None = None, refine_label: bool = True) -> GeometryFit:
    """Fit primitive geometry and optionally override the semantic class by residual."""
    fits = [_cuboid_fit(points), _cylinder_fit(points), _sphere_fit(points)]
    if refine_label:
        return min(fits, key=lambda fit: fit.residual)
    return fits[int(semantic_label)] if semantic_label is not None else min(fits, key=lambda fit: fit.residual)


def recognize_scene(
    points: np.ndarray,
    model: ShapeClassifier,
    *,
    device: str = "auto",
    cluster_eps: float = 0.65,
    min_cluster_points: int = 20,
    refine_geometry: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[RecognizedInstance]]:
    """Recognize separated objects by clustering first, then classifying each instance."""
    pred_labels = np.full(len(points), -1, dtype=np.int64)
    instance_labels = np.full(len(points), -1, dtype=np.int64)
    instance_indices = cluster_spatial_instances(points, eps=cluster_eps, min_points=min_cluster_points)

    instances: list[RecognizedInstance] = []
    for instance_id, idx in enumerate(instance_indices):
        instance_points = points[idx]
        semantic_label, semantic_confidence = classify_instance(model, instance_points, device=device)
        geometry = fit_geometry(instance_points, semantic_label=semantic_label, refine_label=refine_geometry)
        final_label = geometry.label if refine_geometry else semantic_label
        pred_labels[idx] = final_label
        instance_labels[idx] = instance_id
        confidence = min(semantic_confidence, geometry.confidence) if refine_geometry else semantic_confidence
        instances.append(
            RecognizedInstance(
                instance_id=instance_id,
                semantic_label=semantic_label,
                semantic_name=SHAPE_CLASS_NAMES[semantic_label],
                geometry_label=geometry.label,
                geometry_name=geometry.name,
                confidence=confidence,
                num_points=len(idx),
                centroid=instance_points.mean(axis=0).tolist(),
                geometry=geometry,
            )
        )
    return pred_labels, instance_labels, instances


def export_recognition_examples(
    *,
    count: int = 3,
    seed: int = 42,
    output_dir: str | Path = RESULT_DIR / "recognition",
    model_path: str | Path = MODELS_DIR / "shape.pt",
    device: str = "auto",
    cluster_eps: float = 0.65,
    min_cluster_points: int = 20,
    refine_geometry: bool = False,
    num_points: int = 4096,
) -> Path:
    """Generate scenes, run layered recognition, and save JSON summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_shape_classifier(model_path, device=device)
    generator = SceneGenerator()
    summaries = []

    for i in range(count):
        np.random.seed(seed + i)
        points, true_labels = generator.generate_scene_arrays()
        points, true_labels = resample_point_cloud(points, true_labels, num_points)
        pred_labels, instance_labels, instances = recognize_scene(
            points,
            model,
            device=device,
            cluster_eps=cluster_eps,
            min_cluster_points=min_cluster_points,
            refine_geometry=refine_geometry,
        )
        accuracy = float(np.mean(pred_labels == true_labels))
        payload = {
            "scene": i + 1,
            "point_accuracy": accuracy,
            "num_instances": len(instances),
            "instances": [asdict(instance) for instance in instances],
        }
        summaries.append(payload)
        (output_dir / f"scene_{i + 1:02d}.json").write_text(json.dumps(payload, indent=2))
        save_labeled_scene_plot(
            points,
            pred_labels,
            output_dir / f"scene_{i + 1:02d}_pred.png",
            title=f"Scene {i + 1} predicted classes",
        )
        save_labeled_scene_plot(
            points,
            instance_labels,
            output_dir / f"scene_{i + 1:02d}_instances.png",
            title=f"Scene {i + 1} predicted instances",
            use_instance_palette=True,
        )
        print(f"Recognized scene {i + 1}/{count}: {len(instances)} instances, point acc={accuracy:.4f}")

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    return output_dir
