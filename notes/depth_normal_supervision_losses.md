# Depth and Normal Supervision Losses

This note documents the DA3-based geometry supervision added to training: confidence-weighted depth and normal losses against ground-truth maps loaded per camera.

**Implementation:** `utils/loss_utils.py`, `train.py`  
**Related notes:** [depth_colmap_alignment.md](depth_colmap_alignment.md), [depth_to_normals.md](depth_to_normals.md)

---

## Overview

Each training view carries three maps from `frame_data/` (after loading and preprocessing):

| Map | Attribute | Shape | Space / meaning |
|-----|-----------|-------|-----------------|
| Depth | `viewpoint_cam.depth_map` | `(1, H, W)` | Metric camera $z$, COLMAP-aligned |
| Confidence | `viewpoint_cam.confidence_map` | `(1, H, W)` | Per-pixel weight from DA3, min–max normalized to $[0, 1]$ at load |
| Normals | `viewpoint_cam.normal_map` | `(3, H, W)` | Unit normals in **camera space**; border / bad pixels are `NaN` |

Each iteration renders the scene and compares:

| Prediction | Render key | Shape | Space |
|------------|------------|-------|-------|
| Surface depth | `surf_depth` | `(1, H, W)` | Camera $z$ (blend of expected / median depth) |
| Surface normal | `rend_normal` | `(3, H, W)` | **World space** (rasterizer output) |

Both losses use **confidence-weighted** pixel averages and ignore invalid GT pixels.

---

## Confidence Weighting

For per-pixel errors $e_p$ and weights $w_p$, the shared reduction is:

$$
\bar{e} = \begin{cases}
\dfrac{\sum_p w_p \, e_p}{\sum_p w_p} & \text{if } \sum_p w_p > \varepsilon \\
0 & \text{otherwise}
\end{cases}
$$

- Invalid pixels contribute **$e_p = 0$** and **$w_p = 0$** (not `NaN × 0`, which would poison the sum in PyTorch)
- If the weighted sum is non-finite, the loss returns **0**
- $\varepsilon = 10^{-8}$

Implementation: `confidence_weighted_mean()` in `utils/loss_utils.py`.

---

## Depth Loss

**Goal:** Match rendered surface depth to aligned GT depth, emphasizing **nearby** geometry.

**Per-pixel error:** L1

$$
e_p = \left| d_p^{\text{pred}} - d_p^{\text{gt}} \right|
$$

**Weight:** confidence × inverse depth

$$
w_p = c_p \cdot \frac{1}{d_p^{\text{gt}}}
$$

($d^{\text{gt}}$ clamped below at $10^{-6}$ for stability.)

**Valid pixels** (mask $\mathcal{V}$):

- Finite $d^{\text{pred}}$ and $d^{\text{gt}}$
- $d^{\text{gt}} > 10^{-6}$
- $c_p > 0$

Only masked pixels get non-zero error; all others use $e_p = 0$.

**Loss:**

$$
\mathcal{L}_{\text{depth}} = \lambda_{\text{depth}} \cdot \bar{e}_{\text{depth}}
$$

**Predictions / GT:**

- $d^{\text{pred}}$ = `render_pkg["surf_depth"]`
- $d^{\text{gt}}$ = `viewpoint_cam.depth_map`

Function: `depth_supervision_loss()`.

---

## Normal Loss

**Goal:** Match rendered normals to GT normals from DA3 depth.

**Per-pixel error:** angular (for unit vectors)

$$
e_p = 1 - \hat{\mathbf{n}}_p^{\text{pred}} \cdot \hat{\mathbf{n}}_p^{\text{gt}}
$$

Dot product clamped to $[-1, 1]$. Zero error when normals align; 2 when opposite.

**Preprocessing:** Both `pred` and `gt` are **L2-normalized per pixel** (channel dim) before the dot product. GT uses `nan_to_num(..., nan=0.0)` first so border `NaN`s do not propagate into the normalize step (invalid pixels are still excluded by the mask).

**Weight:** $w_p = c_p$

**Valid pixels** (mask $\mathcal{V}$):

- All three GT normal channels finite (excludes `NaN` border / low-magnitude estimates from `depth_to_normal.py`)
- All three predicted channels finite
- $c_p > 0$

Only masked pixels get non-zero error; all others use $e_p = 0$.

**Coordinate frame:** GT normals are stored in **camera space**. Rendered `rend_normal` is in **world space**. Before comparison, GT is transformed with the same convention as the renderer:

$$
\hat{\mathbf{n}}^{\text{world}} = \hat{\mathbf{n}}^{\text{cam}} \, R_{[:3,:3]}^\top
$$

where $R$ is `viewpoint_cam.world_view_transform[:3, :3]`.

Function: `camera_normals_to_world()` then `normal_supervision_loss()`.

**Loss:**

$$
\mathcal{L}_{\text{normal}} = \lambda_{\text{normal}} \cdot \bar{e}_{\text{normal}}
$$

**Predictions / GT:**

- $\hat{\mathbf{n}}^{\text{pred}}$ = `render_pkg["rend_normal"]`
- $\hat{\mathbf{n}}^{\text{gt}}$ = `camera_normals_to_world(normal_map, world_view_transform)`

---

## Training Schedule

Depth and normal supervision are **disabled until** `iteration > iteration_mesh` (default `iteration_mesh = 5000`), same gate as the previous normal regularizer:

```python
lambda_depth  = opt.lambda_depth  if iteration > opt.iteration_mesh else 0
lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
```

Early training relies on photometric loss and densification; geometry supervision kicks in after the mesh / refinement phase.

**Lazy evaluation:** The supervision functions are only called when their $\lambda > 0$. Before `iteration_mesh`, `depth_loss` and `normal_loss` are literal zeros on the render device — they are **not** computed. This avoids ever forming `0 × NaN` (which is `NaN` in PyTorch) from border `NaN`s in `normal_map`.

Progress bar / TensorBoard will show `depth=0`, `normal=0` until supervision activates.

---

## Total Loss

During densification (`iteration < densify_until_iter`):

$$
\mathcal{L} = \mathcal{L}_{\text{image}} + \mathcal{L}_{\text{opacity}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{normal}} + \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{size}}
$$

After densification, $\mathcal{L}_{\text{size}}$ is dropped.

| Term | Description |
|------|-------------|
| $\mathcal{L}_{\text{image}}$ | L1/L2 + DSSIM vs RGB |
| $\mathcal{L}_{\text{opacity}}$ | Opacity regularizer |
| $\mathcal{L}_{\text{depth}}$ | DA3 depth supervision (this note) |
| $\mathcal{L}_{\text{normal}}$ | DA3 normal supervision (this note) |
| $\mathcal{L}_{\text{dist}}$ | Depth distortion (`rend_dist`), optional |
| $\mathcal{L}_{\text{size}}$ | Equilateral regularizer on triangles |

---

## Hyperparameters

| Flag | Default | Role |
|------|---------|------|
| `--lambda_depth` | `0.01` | Scale depth supervision |
| `--lambda_normals` | `0.0001` | Scale normal supervision |
| `--iteration_mesh` | `5000` | First iteration where depth/normal losses are active |

Tune $\lambda_{\text{depth}}$ if near geometry dominates or underfits; tune $\lambda_{\text{normal}}$ if normal artifacts persist in renders.

---

## Logging and Debugging

**Training**

- Progress bar: `depth`, `normal` components (every 10 iters)
- TensorBoard: `train_loss_patches/depth_loss`, `train_loss_patches/normal_loss`

**Pre-training data check**

- `debug_maps.py` — GT RGB, depth, confidence, normals, `depth_scale.txt`

**Post-training render**

- `render.py` → `render_set_extended` saves `normals/` (rendered `rend_normal` visualization) alongside `depth/`, `renders/`, etc.

---

## What Was Replaced

Previously, normal loss compared **rendered** normal vs **pseudo** normal from rendered depth:

```python
# old
normal_error = 1 - (rend_normal * surf_normal).sum(dim=0)
```

That self-consistency term is replaced by direct supervision against DA3 GT normals. Distortion loss (`rend_dist`) is unchanged.

---

## End-to-End Data Flow

```
frame_data/{stem}.npz
  → load depth + confidence
  → resize to train resolution
  → COLMAP scale alignment (depth only)
  → depth_to_normal (bilateral smooth, r=2 FD, → normal_map)
  → stored on Camera

Training step:
  render → surf_depth, rend_normal
  compare to depth_map, camera_normals_to_world(normal_map)
  weight by confidence_map
```

See the linked notes for alignment and normal estimation details.

---

## NaN Safety and Invalid Pixels

GT `normal_map` intentionally contains **`NaN`** on a 2-pixel border (finite-difference radius) and at degenerate estimates. Depth / confidence may also have invalid regions. The loss path handles this as follows:

| Issue | Mitigation |
|-------|------------|
| `NaN` in GT normals | Valid mask requires finite GT; errors zeroed outside mask |
| `NaN × 0` in weighted sum | Errors set to 0 on invalid pixels **before** `(err * w).sum()` |
| Empty valid set (no pixels with $c > 0$) | `confidence_weighted_mean` returns 0 |
| Non-finite weighted sum | Returns 0 |
| $\lambda = 0$ before `iteration_mesh` | Skip loss computation entirely; use scalar 0 |

**Training mask for custom losses:** reuse the same logic or apply `torch.isfinite(normal_map).all(dim=0) & (confidence_map > 0)` on GT maps.

---
