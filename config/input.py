"""User parameters — edit values below, then run: python main.py"""

from dataclasses import dataclass


@dataclass
class Input:
    # --- Pipeline switches ---
    regen: bool = True
    prepare_data: bool = True
    prepare_shape: bool = True
    prepare_scene: bool = True

    train: bool = True
    train_shape: bool = True
    train_scene: bool = True

    preview_shape: bool = True
    preview_scene: bool = True
    export_examples: bool = True
    plot_training_curves: bool = True

    # --- Dataset ---
    num_samples_shape: int = 3000
    num_samples_scene: int = 1000
    num_points_shape: int = 1024
    num_points_scene: int = 4096
    seed: int | None = 42
    cache: bool = True
    preload_workers: int = 0  # 0 = use all CPU cores

    # --- Training ---
    num_epochs_shape: int = 30
    num_epochs_scene: int = 20
    batch_size_shape: int = 16
    batch_size_scene: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0  # DataLoader workers; 0 = auto (cpu_count - 1)
    device: str = "auto"  # "auto", "cuda", or "cpu"

    # --- Training stop (Ctrl+C also saves and exits) ---
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    target_accuracy: float | None = None  # e.g. 0.99 to stop once reached

    # --- Preview / export ---
    scene_preview_count: int = 3
    scene_export_count: int = 3
