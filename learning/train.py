"""PointNet-style models and training loops."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import MODELS_DIR, NUM_SHAPE_CLASSES, RESULT_DIR, configure_cpu_threads, resolve_device, resolve_num_workers, resolve_preload_workers
from config.input import Input
from learning.datasets import SceneDataset, ShapeDataset
from learning.models import SceneSegmenter, ShapeClassifier
from simulation.generation import normalize_point_cloud

TaskKind = Literal["shape", "scene"]


@dataclass
class EarlyStopState:
    best_acc: float = -1.0
    best_loss: float = float("inf")
    best_epoch: int = -1
    patience: int = 0
    reason: str = ""

    def history_entry(self, epoch: int, loss: float, acc: float, improved: bool) -> dict:
        return {
            "epoch": epoch,
            "loss": loss,
            "acc": acc,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_acc": self.best_acc,
            "improved": improved,
            "patience": self.patience,
        }


def _setup_device(device: str, on_the_fly: bool, num_workers: int, cpu_threads: int) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable, falling back to CPU.")
        device = "cpu"

    mode = "on-the-fly" if on_the_fly else "from data/"
    if device == "cuda":
        print(f"Training on: cuda ({torch.cuda.get_device_name(0)}) | data: {mode}")
    elif device == "mps":
        print(f"Training on: mps (Apple GPU) | data: {mode}, loader workers: {num_workers}")
    else:
        threads = configure_cpu_threads(num_workers, cpu_threads)
        print(f"Training on: cpu ({threads} threads) | data: {mode}, loader workers: {num_workers}")
    return device


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()


def _make_loader(dataset, batch_size: int, num_workers: int, device: str) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _scene_class_weights(dataset, device: str) -> torch.Tensor | None:
    cache = getattr(dataset, "_cache", None)
    if cache is None:
        return None
    counts = torch.zeros(NUM_SHAPE_CLASSES, dtype=torch.float32)
    for _, labels in cache.values():
        counts += torch.bincount(torch.as_tensor(labels), minlength=NUM_SHAPE_CLASSES).float()
    weights = counts.sum() / (NUM_SHAPE_CLASSES * counts.clamp_min(1.0))
    return weights.to(device)


def _save_model_state(state: dict[str, torch.Tensor], name: str, *, best: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    torch.save(state, path)
    prefix = "Best model" if best else "Model"
    print(f"{prefix} saved to {path}")
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


def _should_stop_training(inp: Input, loss: float, acc: float, epoch: int, state: EarlyStopState) -> tuple[bool, EarlyStopState, bool]:
    acc_improved = acc > state.best_acc + inp.early_stop_min_delta
    loss_improved = loss < state.best_loss - inp.early_stop_loss_min_delta
    improved = acc_improved or (abs(acc - state.best_acc) <= inp.early_stop_min_delta and loss_improved)

    if improved:
        state.best_acc = max(state.best_acc, acc)
        state.best_loss = min(state.best_loss, loss)
        state.best_epoch = epoch
        state.patience = 0
    else:
        state.patience += 1

    if not inp.early_stop:
        return False, state, improved

    if epoch + 1 < inp.early_stop_min_epochs:
        return False, state, improved

    if inp.target_accuracy is not None and state.best_acc >= inp.target_accuracy:
        state.reason = f"target accuracy {inp.target_accuracy:.4f} reached"
        print(f"Early stopping: {state.reason}.")
        return True, state, improved

    if state.patience >= inp.early_stop_patience:
        state.reason = f"no accuracy/loss improvement for {inp.early_stop_patience} epochs"
        print(f"Early stopping: {state.reason}.")
        return True, state, improved
    return False, state, improved


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
    device = _setup_device(resolve_device(inp.device), dataset.on_the_fly, num_workers, inp.cpu_threads)
    loader = _make_loader(dataset, batch_size, num_workers, device)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=inp.lr, weight_decay=inp.weight_decay)
    class_weights = _scene_class_weights(dataset, device) if task == "scene" else None
    if class_weights is not None:
        print(f"Scene class weights: {[round(float(w), 4) for w in class_weights.cpu()]}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"=== {log_prefix} ===")
    history: list[dict] = []
    stop_state = EarlyStopState()
    best_model_state: dict[str, torch.Tensor] | None = None
    status = "completed"
    interrupted = False

    try:
        for epoch in range(num_epochs):
            loss, acc = _run_epoch(model, loader, optimizer, loss_fn, device, task)
            should_stop, stop_state, improved = _should_stop_training(inp, loss, acc, epoch, stop_state)
            if improved:
                best_model_state = deepcopy(model.state_dict())
            history.append(stop_state.history_entry(epoch, loss, acc, improved))
            marker = " *best*" if improved else ""
            print(
                f"[{log_prefix} | Epoch {epoch}] "
                f"Loss: {loss:.4f} | Acc: {acc:.4f} | "
                f"Best Acc: {stop_state.best_acc:.4f} @ {stop_state.best_epoch}{marker}"
            )

            if should_stop:
                status = "early_stopped"
                break
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        print("\nTraining interrupted. Saving current progress...")
    finally:
        if history:
            if best_model_state is not None:
                _save_model_state(best_model_state, model_name, best=True)
            else:
                _save_model_state(model.state_dict(), model_name)
            _append_result_log(stage, history, status)
            if stop_state.reason:
                print(f"{log_prefix} finished ({status}: {stop_state.reason}).")
            else:
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
        build_model=lambda: SceneSegmenter(num_classes=NUM_SHAPE_CLASSES, k_neighbors=inp.scene_k_neighbors),
    )
