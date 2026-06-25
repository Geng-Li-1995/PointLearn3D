# PointLearn3D

[![CI](https://github.com/Geng-Li-1995/PointLearn3D/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/PointLearn3D/actions/workflows/ci.yml)

Procedural 3D scene simulation and PointNet-style point cloud learning for **cuboids**, **cylinders**, and **spheres**.

An end-to-end pipeline: synthetic geometry → NPZ dataset cache → shape classification & scene segmentation → Open3D previews and training-curve plots. Configure everything in `config/input.py`, then run `python main.py`.

**Author:** [Dr. Geng Li](https://github.com/Geng-Li-1995)

---

## Features

| Module | Description |
|--------|-------------|
| `simulation/generation.py` | Parametric primitives, voxel collision, `ShapeGenerator` / `SceneGenerator` |
| `learning/datasets.py` | `ShapeDataset` / `SceneDataset` with NPZ cache and parallel preloading |
| `learning/train.py` | `ShapeClassifier`, `SceneSegmenter`, early stopping, Ctrl+C checkpointing |
| `learning/visualize.py` | Open3D previews, PNG export, training curves |
| `config/input.py` | Switches, hyperparameters, and training stop criteria |

---

## Project layout

```
PointLearn3D/
├── main.py
├── config/
│   ├── input.py             # User-editable parameters
│   └── config.py            # Paths, constants, run()
├── simulation/generation.py
├── learning/
│   ├── datasets.py
│   ├── train.py
│   ├── visualize.py
│   └── plot_set.py
├── tests/                   # pytest (22 tests)
├── .github/workflows/ci.yml # GitHub Actions
├── data/                    # NPZ caches (git-ignored)
├── result/                  # Models, logs, figures (see below)
└── requirements.txt
```

---

## Quick start

**Requirements:** Python 3.10+ · PyTorch · NumPy · Matplotlib · Open3D (`requirements.txt`)

```bash
git clone https://github.com/Geng-Li-1995/PointLearn3D.git
cd PointLearn3D
pip install -r requirements.txt
python main.py
```

Edit **`config/input.py`** before running. There is no CLI `--mode`.

| Step | Switch | Effect |
|------|--------|--------|
| 1 | `preview_shape` / `preview_scene` | Interactive Open3D windows |
| 2 | `export_examples` | PNGs → `result/shape/`, `result/scene/` |
| 3 | `prepare_data` | Build `data/shape/`, `data/scene/` caches |
| 4 | `train_shape` / `train_scene` | Weights → `result/models/` |
| 5 | `plot_training_curves` | Plots → `result/training/` |

**Git:** only `data/*.npz` is ignored; `result/` outputs can be committed.

### Tests & CI

```bash
pytest          # local
pytest -v tests/test_train.py
```

[![CI](https://github.com/Geng-Li-1995/PointLearn3D/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/PointLearn3D/actions/workflows/ci.yml) runs on every push/PR to `main` / `master` (Ubuntu, Python 3.10–3.12). Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Visualizations

Figures below live under `result/` (from `export_examples` and `plot_training_curves`). Re-run `python main.py` to refresh.

### Single-shape point clouds

Red = cuboid · Green = cylinder · Blue = sphere

| Cuboid | Cylinder | Sphere |
|:------:|:--------:|:------:|
| ![cuboid](result/shape/cuboid.png) | ![cylinder](result/shape/cylinder.png) | ![sphere](result/shape/sphere.png) |

### Multi-object scenes

Voxel-grid placement; color = object class

| Scene 1 | Scene 2 |
|:-------:|:-------:|
| ![scene 1](result/scene/scene_01.png) | ![scene 2](result/scene/scene_02.png) |

### Training curves

**Shape classification** — ~96% accuracy after early stopping:

![shape training curves](result/training/shape_curves.png)

**Scene segmentation** — baseline; still needs improvement ([limitations](#known-limitations)):

![scene training curves](result/training/scene_curves.png)

---

## Pipeline

`config/config.py` → `run()` executes in order:

| # | Switch | Action |
|---|--------|--------|
| 1 | `preview_shape` | Open3D: cuboid, cylinder, sphere |
| 2 | `preview_scene` | Open3D: `scene_preview_count` scenes |
| 3 | `export_examples` | PNG export |
| 4 | `prepare_data` | NPZ cache build (`regen` to rebuild) |
| 5 | `train` | Model training |
| 6 | `plot_training_curves` | Loss / accuracy plots |

---

## Training

### Shape classification (`train_shape`)

| | |
|---|---|
| Input | Single-object point cloud |
| Model | `ShapeClassifier` (PointNet-style global max-pool + MLP) |
| Weights | `result/models/shape.pt` |
| Log key | `shape` |

### Scene segmentation (`train_scene`)

| | |
|---|---|
| Input | Multi-object scene, per-point labels |
| Model | `SceneSegmenter` (independent weights) |
| Weights | `result/models/scene.pt` |
| Log key | `scene` |

> **Work in progress.** `SceneSegmenter` is a lightweight baseline (global max-pool + broadcast). No full local–global fusion or PointNet++-style backbone yet. Segmentation on crowded scenes may be weak.

Metrics → `result/training_log.json` (latest run per stage used for plots).

---

## Scene generation

`SceneGenerator` places random primitives in a bounded volume. Overlap checks use a **voxel grid** (`VoxelEngine`), not a KD-tree. Each object is resampled to a fixed point count with a class label.

---

## Configuration (`config/input.py`)

| Group | Fields |
|-------|--------|
| Switches | `regen`, `prepare_data`, `prepare_shape`, `prepare_scene`, `train`, `train_shape`, `train_scene`, `preview_*`, `export_examples`, `plot_training_curves` |
| Dataset | `num_samples_shape`, `num_samples_scene`, `num_points_shape`, `num_points_scene`, `seed`, `cache`, `preload_workers` |
| Training | `num_epochs_shape`, `num_epochs_scene`, `batch_size_shape`, `batch_size_scene`, `lr`, `weight_decay`, `num_workers`, `device` |
| Early stop | `early_stop`, `early_stop_patience`, `early_stop_min_delta`, `target_accuracy` |
| Preview | `scene_preview_count`, `scene_export_count` |

- `preload_workers=0` → all CPU cores for preloading  
- `num_workers=0` → DataLoader uses `cpu_count - 1`  
- Same-run `prepare_data` + `train` → training skips rebuild (`regen=False`)  
- **Ctrl+C** saves weights and log for the current epoch  

---

## Outputs

| Path | Contents |
|------|----------|
| `data/shape/dataset.npz` | Shape cache (git-ignored) |
| `data/scene/dataset.npz` | Scene cache (git-ignored) |
| `result/models/shape.pt`, `scene.pt` | Weights |
| `result/training_log.json` | Per-epoch metrics |
| `result/training/*_curves.png` | Training curves |
| `result/shape/`, `result/scene/` | Example PNGs |

---

## Known limitations

| Area | Status |
|------|--------|
| Shape classification | Stable baseline for single primitives |
| Scene segmentation | **Needs optimization** |
| Scene generation | Voxel placement; no physics or realistic occlusion |

---

## Author

**Dr. Geng Li** — theoretical physics & lattice QCD background; scientific computing, HPC, and ML on structured data.

---

## License

No license specified. Contact the author before redistribution or commercial use.
