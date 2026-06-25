"""Matplotlib static plots and Open3D interactive previews."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import open3d as o3d

from config.config import (
    CLASS_COLORS,
    DEFAULT_COLOR,
    RESULT_DIR,
    RESULT_SCENE_DIR,
    RESULT_SHAPE_DIR,
    SHAPE_CLASS_NAMES,
    TRAINING_PLOTS_DIR,
    TRAINING_STAGE_LABELS,
)
from learning.plot_set import (
    FIG_METRICS,
    FS_SUPTITLE,
    apply_plot_style,
    plot_series,
    save_figure,
    style_axes,
)
from simulation.generation import SceneGenerator, ShapeGenerator

def _create_pcd(points, color=DEFAULT_COLOR):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def _set_equal_aspect(ax, points: np.ndarray, pad: float = 0.05) -> None:
    """Cube limits from max radial extent (good for single shapes)."""
    center = points.mean(axis=0)
    radius = max(np.max(np.linalg.norm(points - center, axis=1)), 1e-3)
    radius *= 1 + pad
    for axis, c in zip("xyz", center):
        getattr(ax, f"set_{axis}lim")(c - radius, c + radius)


def _set_scene_view(ax, points: np.ndarray, pad: float = 0.06) -> None:
    """Tight per-axis limits with box aspect matching data spans (no stretch, minimal z padding)."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    margin = spans * pad
    for i, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(mins[i] - margin[i], maxs[i] + margin[i])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(tuple(spans))


def _crop_white_margins(path: Path, pad_px: int = 10, thresh: float = 0.97) -> None:
    """Trim near-white borders from a saved PNG (matplotlib 3D leaves large margins)."""
    data = mpimg.imread(path)
    rgb = data[..., :3] if data.ndim == 3 else data
    content = np.any(rgb < thresh, axis=-1)
    if not content.any():
        return
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    r0, r1 = max(0, rows[0] - pad_px), min(data.shape[0], rows[-1] + pad_px + 1)
    c0, c1 = max(0, cols[0] - pad_px), min(data.shape[1], cols[-1] + pad_px + 1)
    mpimg.imsave(path, data[r0:r1, c0:c1])


def _plot_point_cloud(ax, points: np.ndarray, color: tuple[float, float, float], size: float = 1.0):
    color_arr = np.broadcast_to(color, (len(points), 3))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_arr, s=size, linewidths=0, alpha=0.9)
    _set_equal_aspect(ax, points)


def _save_3d_figure(fig, ax, path: Path, dpi: int, *, top: float = 0.92) -> None:
    """Tight export for 3D axes (matplotlib leaves large margins by default)."""
    ax.set_position([0, 0, 1, top])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=top)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")


def save_scene_plot(objects, path: str | Path, title: str | None = None, dpi: int = 150) -> None:
    """Save a single scene as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    all_points = []
    for obj in objects:
        color_arr = np.broadcast_to(CLASS_COLORS.get(obj.label, DEFAULT_COLOR), (len(obj.points), 3))
        ax.scatter(
            obj.points[:, 0], obj.points[:, 1], obj.points[:, 2],
            c=color_arr, s=1.2, linewidths=0, alpha=0.9,
        )
        all_points.append(obj.points)
    if all_points:
        _set_scene_view(ax, np.concatenate(all_points, axis=0))
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()
    ax.dist = 4
    if title:
        ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="none", pad_inches=0)
    plt.close(fig)
    _crop_white_margins(path, pad_px=6)


def save_shape_plot(points: np.ndarray, label: int, path: str | Path, title: str | None = None, dpi: int = 150):
    """Save a single shape point cloud as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _plot_point_cloud(ax, points, CLASS_COLORS.get(label, DEFAULT_COLOR), size=2.0)
    ax.view_init(elev=20, azim=-65)
    ax.set_axis_off()
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))
    ax.set_title(title or SHAPE_CLASS_NAMES.get(label, "unknown"), fontsize=11, pad=4)
    _save_3d_figure(fig, ax, path, dpi, top=0.90)
    plt.close(fig)


def visualize_scene(objects, labels=None):
    """Open an interactive Open3D window for a scene."""
    geoms = []
    for i, obj in enumerate(objects):
        label = labels[i] if labels is not None else obj.label
        geoms.append(_create_pcd(obj.points, CLASS_COLORS.get(label, DEFAULT_COLOR)))
    o3d.visualization.draw_geometries(geoms)


def visualize_single_object(points, label=None):
    """Open an interactive Open3D window for one point cloud."""
    color = CLASS_COLORS.get(label, DEFAULT_COLOR) if label is not None else DEFAULT_COLOR
    o3d.visualization.draw_geometries([_create_pcd(points, color)])


def preview_shapes(n_points: int = 1024, seed: int | None = None) -> None:
    """Open Open3D windows for cuboid, cylinder, and sphere (one each)."""
    gen = ShapeGenerator(n_points=n_points)
    for label, name in SHAPE_CLASS_NAMES.items():
        if seed is not None:
            np.random.seed(seed + label)
        points, _ = gen.generate(shape_class=label)
        print(f"Preview shape: {name} (class {label}), points: {points.shape}")
        visualize_single_object(points, label=label)


def preview_scenes(count: int = 3, seed: int | None = None) -> None:
    """Open Open3D windows for multiple generated scenes."""
    gen = SceneGenerator()
    for i in range(count):
        if seed is not None:
            np.random.seed(seed + i)
        objects = gen.generate_scene()
        print(f"Preview scene {i + 1}/{count}: {len(objects)} objects")
        visualize_scene(objects)


def export_visualization_examples(
    output_dir: str | Path = RESULT_DIR, scene_count: int = 3, seed: int = 42,
) -> Path:
    """Generate sample shape and scene PNGs under result/."""
    output_dir = Path(output_dir)
    scene_dir = RESULT_SCENE_DIR if output_dir == RESULT_DIR else output_dir / "scene"
    shape_dir = RESULT_SHAPE_DIR if output_dir == RESULT_DIR else output_dir / "shape"
    scene_dir.mkdir(parents=True, exist_ok=True)
    shape_dir.mkdir(parents=True, exist_ok=True)

    scene_gen = SceneGenerator()
    shape_gen = ShapeGenerator(n_points=1024)

    for i in range(scene_count):
        np.random.seed(seed + i)
        objects = scene_gen.generate_scene()
        out_path = scene_dir / f"scene_{i + 1:02d}.png"
        save_scene_plot(objects, out_path, title=f"Scene {i + 1} ({len(objects)} objects)")
        print(f"Saved {out_path}")

    for label, name in SHAPE_CLASS_NAMES.items():
        np.random.seed(seed + 100 + label)
        points, _ = shape_gen.generate(shape_class=label)
        out_path = shape_dir / f"{name}.png"
        save_shape_plot(points, label, out_path)
        print(f"Saved {out_path}")

    return output_dir


def _load_latest_runs(log_path: Path) -> dict[str, dict]:
    """Return the most recent log entry per training stage."""
    logs = json.loads(log_path.read_text())
    latest: dict[str, dict] = {}
    for entry in logs:
        latest[entry["stage"]] = entry
    return latest


def _plot_stage_curves(ax_loss, ax_acc, entry: dict, title: str, color_idx: int = 0) -> None:
    epochs = np.array([e["epoch"] for e in entry["epochs"]])
    losses = np.array([e["loss"] for e in entry["epochs"]])
    accs = np.array([e["acc"] for e in entry["epochs"]])

    plot_series(ax_loss, epochs, losses, color_idx=color_idx, label="Loss")
    style_axes(ax_loss, xlabel="Epoch", ylabel="Loss", title=f"{title} — Loss")
    ax_loss.legend(loc="best", framealpha=0.95)

    plot_series(ax_acc, epochs, accs, color_idx=color_idx + 1, label="Accuracy")
    style_axes(ax_acc, xlabel="Epoch", ylabel="Accuracy", title=f"{title} — Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.legend(loc="best", framealpha=0.95)


def plot_training_curves(
    log_path: str | Path = RESULT_DIR / "training_log.json",
    output_dir: str | Path = TRAINING_PLOTS_DIR,
    dpi: int = 150,
) -> Path | None:
    """Plot loss and accuracy curves from training_log.json into result/training/."""
    apply_plot_style()
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"No training log at {log_path}, skipping training curve plots.")
        return None

    latest = _load_latest_runs(log_path)
    if not latest:
        print("Training log is empty, skipping training curve plots.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (stage, entry) in enumerate(latest.items()):
        title = TRAINING_STAGE_LABELS.get(stage, stage)
        status = entry.get("status", "completed")
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=FIG_METRICS)
        _plot_stage_curves(ax_loss, ax_acc, entry, title, color_idx=i * 2)
        fig.suptitle(f"{title}  ({status}, {entry['updated_at']})", fontsize=FS_SUPTITLE, y=1.02)
        out_path = save_figure(fig, output_dir / f"{stage}_curves.png", dpi=dpi)
        plt.close(fig)
        print(f"Saved {out_path}")

    return output_dir
