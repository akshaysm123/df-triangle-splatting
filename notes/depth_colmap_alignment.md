# Depth Map Alignment to COLMAP Scale

This note documents how predicted depth maps (e.g. from DepthAnything) are scaled to match COLMAP’s metric coordinate system before training.

**Implementation:** `utils/depth_colmap_align.py`, called from `utils/camera_utils.py` (`loadCam`).

---

## Problem

Monocular depth predictors typically return depth up to an unknown global scale (and sometimes shift). COLMAP reconstruction uses a consistent metric frame with sparse 3D points and calibrated cameras. To use predicted depth for supervision or geometry, we estimate a **per-image scalar scale** that aligns predicted depth with COLMAP.

We assume a **pure scale** ambiguity (no per-pixel offset):

\[
d^{\text{metric}} \approx s \cdot d^{\text{pred}}
\]

---

## Data Sources

| Source | Location | Used for |
|--------|----------|----------|
| Predicted depth | `{scene}/frame_data/{image_stem}.npz` → key `"depth"` | \(d^{\text{pred}}\) |
| COLMAP cameras | `{scene}/sparse/0/images.{bin,txt}` | Pose, 2D observations |
| COLMAP intrinsics | `{scene}/sparse/0/cameras.{bin,txt}` | Image size for pixel scaling |
| COLMAP points | `{scene}/sparse/0/points3D.{bin,txt}` | 3D positions \(\mathbf{X}\) |

Depth and confidence live in the **same** `.npz` file; only `"depth"` is used for alignment. Confidence is not scaled.

---

## Correspondences

For each COLMAP image, every 2D feature with a valid 3D point ID gives one correspondence:

- **Pixel** \((u, v)\): COLMAP observation `xys` (in intrinsics image coordinates).
- **3D point** \(\mathbf{X}\): world coordinates from `points3D`.
- **Predicted depth** \(d^{\text{pred}}\): value at the corresponding pixel in the resized depth map.

Points with `point3D_id < 0` are skipped.

---

## COLMAP Depth in Camera Space

COLMAP uses world-to-camera transform \(\mathbf{X}_{\text{cam}} = R \mathbf{X} + \mathbf{t}\).

In this codebase, cameras store `R` as the **transpose** of COLMAP’s rotation matrix (`R_{\text{stored}} = R^\top`). The COLMAP rotation is recovered as:

\[
R_{\text{colmap}} = R_{\text{stored}}^\top
\]

\[
\mathbf{X}_{\text{cam}} = R_{\text{colmap}} \mathbf{X} + \mathbf{t}
\]

**COLMAP depth** (positive forward depth along camera \(z\)):

\[
z^{\text{colmap}} = \mathbf{X}_{\text{cam}, z}
\]

---

## Pixel Coordinates and Resizing

Depth maps are resized to **training resolution** \((W_t, H_t)\) before alignment (see `utils/DA3_utils.py`: nearest-neighbor resize to full RGB size, then to train size).

COLMAP \((u, v)\) are defined in **intrinsics** resolution \((W_c, H_c)\). They are mapped to training pixels via:

\[
u_t = \mathrm{round}\!\left(u \cdot \frac{W_t}{W_c}\right), \qquad
v_t = \mathrm{round}\!\left(v \cdot \frac{H_t}{H_c}\right)
\]

Predicted depth is sampled at integer indices \((u_t, v_t)\) on the training-resolution map (nearest / rounded lookup).

---

## Least-Squares Scale (Through the Origin)

For \(N\) valid pairs \((d_i^{\text{pred}}, z_i^{\text{colmap}})\), find \(s\) minimizing:

\[
\sum_{i=1}^{N} \left( s \, d_i^{\text{pred}} - z_i^{\text{colmap}} \right)^2
\]

This is scalar linear regression through the origin. The closed-form solution:

\[
s^* = \frac{\sum_i d_i^{\text{pred}} \, z_i^{\text{colmap}}}{\sum_i \left(d_i^{\text{pred}}\right)^2}
             = \frac{\mathbf{d}^\top \mathbf{z}}{\mathbf{d}^\top \mathbf{d}}
\]

**Aligned depth:**

\[
d^{\text{aligned}} = s^* \cdot d^{\text{pred}}
\]

---

## Validity Filters

A correspondence is used only if:

- `point3D_id ≥ 0` and point exists in the reconstruction
- \(z^{\text{colmap}} > \epsilon\) and finite
- \((u_t, v_t)\) inside the training image bounds
- \(d^{\text{pred}} > \epsilon\) and finite at \((v_t, u_t)\)

If fewer than **10** valid pairs remain (`min_pairs`), \(s^* = 1\) (no scaling) for that view.

---

## Pipeline Order (per camera)

1. Load raw depth from `frame_data/{stem}.npz`
2. Resize to RGB size, then to training resolution (`prepare_map_for_camera`)
3. Estimate \(s^*\) and multiply depth by \(s^*\) (`align_depth_to_colmap`)
4. Compute normals from aligned depth (`depth_map_to_normals` — see `notes/depth_to_normals.md`)

The scale factor is stored on each `Camera` as `depth_scale`.

---

## Coordinate Convention Summary

- **COLMAP camera frame:** \(+X\) right, \(+Y\) down, \(+Z\) forward (undistorted pinhole).
- **Depth value:** \(z\) in camera space after alignment, consistent with COLMAP sparse points.
- **Scope:** Per-view scale; no global scale shared across all images in the current implementation.

---

## Debugging

- Run `debug_maps.py` to export RGB, depth, confidence, normals, and `depth_scale.txt` per view.
- Log line at load time reports min / max / mean `depth_scale` across training cameras.
