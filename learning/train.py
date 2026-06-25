"""PointNet-style models and training loops."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.config import MODELS_DIR, NUM_SHAPE_CLASSES, RESULT_DIR, configure_cpu_threads, resolve_device, resolve_num_workers, resolve_preload_workers
from config.input import Input
from learning.datasets import SceneDataset, ShapeDataset
from simulation.generation import normalize_point_cloud

TaskKind = Literal["shape", "scene"]


class LocalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, out_dim)
        self.mlp2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_points = x.shape[1]
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x.repeat(1, num_points, 1)


class PointFeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.local1 = LocalBlock(3, 64)
        self.local2 = LocalBlock(64, 128)
        self.local3 = LocalBlock(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local3(self.local2(self.local1(x)))


class ShapeClassifier(nn.Module):
    """Classify an entire point cloud as cuboid / cylinder / sphere."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = PointFeatureBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(torch.max(features, dim=1)[0])


class SceneSegmenter(nn.Module):
    """Assign a shape label to every point in a multi-object scene."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = PointFeatureBackbone()
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _setup_device(device: str, on_the_fly: bool, num_workers: int) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    loader_workers = num_workers if on_the_fly else 0
    if device == "cuda":
        mode = "on-the-fly" if on_the_fly else "from data/"
        print(f"Training on: cuda ({torch.cuda.get_device_name(0)}) | data: {mode}")
    else:
        threads = configure_cpu_threads(loader_workers)
        mode = "on-the-fly" if on_the_fly else "from data/"
        print(f"Training on: cpu ({threads} threads) | data: {mode}, loader workers: {loader_workers}")
    return device


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()


def _make_loader(dataset, batch_size: int, num_workers: int, device: str) -> DataLoader:
    loader_workers = num_workers if dataset.on_the_fly else 0
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=loader_workers,
        pin_memory=device == "cuda",
    )
    if loader_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _save_model(model: nn.Module, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


def _append_result_log(stage: str, history: list[dict], status: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_DIR / "training_log.json"
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "epochs": history,
    }
    logs = json.loads(log_path.read_text()) if log_path.exists() else []
    logs.append(payload)
    log_path.write_text(json.dumps(logs, indent=2))


def _run_epoch(model, loader, optimizer, loss_fn, device, task: TaskKind):
    model.train()
    total_loss = total_acc = 0.0
    num_batches = 0
    for points, labels in loader:
        points = normalize_point_cloud(points.to(device))
        labels = labels.to(device)
        logits = model(points)
        if task == "shape":
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
        else:
            loss = loss_fn(logits.reshape(-1, NUM_SHAPE_CLASSES), labels.reshape(-1))
            preds = torch.argmax(logits, dim=-1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_acc += _accuracy(preds, labels)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


def _should_stop_training(inp: Input, acc: float, best_acc: float, patience: int) -> tuple[bool, float, int]:
    if inp.target_accuracy is not None and acc >= inp.target_accuracy:
        print(f"Target accuracy {inp.target_accuracy:.4f} reached.")
        return True, best_acc, patience

    if not inp.early_stop:
        return False, max(best_acc, acc), patience

    if acc > best_acc + inp.early_stop_min_delta:
        return False, acc, 0

    patience += 1
    if patience >= inp.early_stop_patience:
        print(f"Early stopping: no improvement for {inp.early_stop_patience} epochs.")
        return True, best_acc, patience
    return False, best_acc, patience


def _run_training(
    *,
    inp: Input,
    stage: str,
    model_name: str,
    task: TaskKind,
    log_prefix: str,
    num_epochs: int,
    batch_size: int,
    regen: bool,
    build_dataset: Callable,
    build_model: Callable[[], nn.Module],
) -> nn.Module:
    num_workers = resolve_num_workers(inp.num_workers)
    dataset = build_dataset(regen)
    device = _setup_device(resolve_device(inp.device), dataset.on_the_fly, num_workers)
    loader = _make_loader(dataset, batch_size, num_workers, device)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=inp.lr, weight_decay=inp.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print(f"=== {log_prefix} ===")
    history: list[dict] = []
    best_acc = -1.0
    patience = 0
    status = "completed"
    interrupted = False

    try:
        for epoch in range(num_epochs):
            loss, acc = _run_epoch(model, loader, optimizer, loss_fn, device, task)
            history.append({"epoch": epoch, "loss": loss, "acc": acc})
            print(f"[{log_prefix} | Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4f}")

            should_stop, best_acc, patience = _should_stop_training(inp, acc, best_acc, patience)
            if should_stop:
                status = "early_stopped"
                break
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        print("\nTraining interrupted. Saving current progress...")
    finally:
        if history:
            _save_model(model, model_name)
            _append_result_log(stage, history, status)
            print(f"{log_prefix} finished ({status}).")
        elif interrupted:
            print(f"{log_prefix} stopped before completing an epoch.")

    return model


def train_shape(inp: Input, *, regen: bool | None = None) -> nn.Module:
    effective_regen = inp.regen if regen is None else regen
    preload_workers = resolve_preload_workers(inp.preload_workers)

    def build_dataset(r: bool):
        return ShapeDataset(
            num_samples=inp.num_samples_shape,
            num_points=inp.num_points_shape,
            cache=inp.cache,
            seed=inp.seed,
            regen=r,
            preload_workers=preload_workers,
        )

    return _run_training(
        inp=inp,
        stage="shape",
        model_name="shape",
        task="shape",
        log_prefix="Shape classification",
        num_epochs=inp.num_epochs_shape,
        batch_size=inp.batch_size_shape,
        regen=effective_regen,
        build_dataset=build_dataset,
        build_model=lambda: ShapeClassifier(num_classes=NUM_SHAPE_CLASSES),
    )


def train_scene(inp: Input, *, regen: bool | None = None) -> nn.Module:
    effective_regen = inp.regen if regen is None else regen
    preload_workers = resolve_preload_workers(inp.preload_workers)

    def build_dataset(r: bool):
        return SceneDataset(
            num_samples=inp.num_samples_scene,
            num_points=inp.num_points_scene,
            cache=inp.cache,
            seed=inp.seed,
            regen=r,
            preload_workers=preload_workers,
        )

    return _run_training(
        inp=inp,
        stage="scene",
        model_name="scene",
        task="scene",
        log_prefix="Scene segmentation",
        num_epochs=inp.num_epochs_scene,
        batch_size=inp.batch_size_scene,
        regen=effective_regen,
        build_dataset=build_dataset,
        build_model=lambda: SceneSegmenter(num_classes=NUM_SHAPE_CLASSES),
    )
