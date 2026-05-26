# Surface Normals from Depth Maps

This note documents how unit surface normals are estimated from aligned depth maps in **camera space**, using a regular grid of backprojected 3D points and finite-difference tangents.

**Implementation:** `utils/depth_to_normal.py` (`depth_map_to_normals`), called from `utils/camera_utils.py` after COLMAP depth scaling.

---

## Inputs and Outputs

| Item | Shape | Description |
|------|-------|-------------|
| Input `depth` | `(1, H, W)` or `(H, W)` | Aligned metric depth; $z = d$ in camera frame |
| Intrinsics | scalars `fx, fy, cx, cy` | Pinhole parameters in **pixels** at training resolution |
| Output `normal_map` | `(3, H, W)` | Unit normals in camera space; borders / invalid → `NaN` |

Focal lengths are derived from FoV and image size:

$$
f_x = \frac{W}{2 \tan(\mathrm{FoV}_x / 2)}, \qquad
f_y = \frac{H}{2 \tan(\mathrm{FoV}_y / 2)}
$$

Principal point defaults to image center: $c_x = W/2$, $c_y = H/2$.

---

## Step 0: Depth Smoothing (optional, on by default)

Before backprojection, depth is smoothed on CPU with OpenCV:

```python
cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=5)
```

This reduces high-frequency noise in the depth map that would otherwise create spiky normal artifacts. Disable via `smooth_depth=False` in `depth_map_to_normals`.

---

## Step 1: Backproject to Camera-Space Points

For each pixel $(u, v)$ with depth $d(u,v)$, the 3D point in **camera coordinates** is:

$$
\mathbf{P}(u,v) = d \cdot K^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
$$

With standard pinhole $K$:

$$
P_x = \frac{(u - c_x)\, d}{f_x}, \qquad
P_y = \frac{(v - c_y)\, d}{f_y}, \qquad
P_z = d
$$

This yields an $H \times W \times 3$ **structured point cloud**—one point per pixel, row index $v$, column index $u$.

---

## Step 2: Tangent Vectors via Central Differences

Because neighbors lie on a grid, tangents use **central differences** with radius $r$ (default $r = 2$) on interior pixels:

**Along $u$ (horizontal):**

$$
\mathbf{t}_x(u,v) = \mathbf{P}(u+r, v) - \mathbf{P}(u-r, v)
$$

**Along $v$ (vertical):**

$$
\mathbf{t}_y(u,v) = \mathbf{P}(u, v+r) - \mathbf{P}(u, v-r)
$$

Let `span = 2r`. In tensor layout `points[v, u, :]` (row = $v$, col = $u$):

- $\mathbf{t}_x$ = `points[:, span:, :] - points[:, :-span, :]` → shape $(H,\, W - 2r,\, 3)$
- $\mathbf{t}_y$ = `points[span:, :, :] - points[:-span, :, :]` → shape $(H - 2r,\, W,\, 3)$

Cross product on the **interior overlap** (both tangents defined at the same center $(u,v)$):

- `t_x_mid = t_x[r:-r, :, :]` → $(H - 2r,\, W - 2r,\, 3)$
- `t_y_mid = t_y[:, r:-r, :]` → $(H - 2r,\, W - 2r,\, 3)$

Valid centers: $u \in [r,\, W-r-1]$, $v \in [r,\, H-r-1]$. With $r=2$, the **border is 2 pixels** wide (all `NaN`).

---

## Step 3: Normal via Cross Product

Unnormalized normal (interior pixels only):

$$
\mathbf{n}(u,v) = \mathbf{t}_x(u,v) \times \mathbf{t}_y(u,v)
$$

Let $m = \|\mathbf{n}\|$. If $m < \tau$ (default $\tau = 10^{-4}$), the pixel is set to **`NaN`** (degenerate / near-flat patch). Otherwise:

$$
\hat{\mathbf{n}}(u,v) = \frac{\mathbf{n}}{m}
$$

---

## Step 4: Face the Camera

The camera is at the origin in camera space. The **view direction** toward the surface point is:

$$
\mathbf{v}(u,v) = \mathbf{P}(u,v)
$$

Normals should point **toward** the camera (visible front face). If the normal faces away:

$$
\hat{\mathbf{n}} \cdot \mathbf{P} > 0 \quad \Rightarrow \quad \hat{\mathbf{n}} \leftarrow -\hat{\mathbf{n}}
$$

Equivalently: keep the normal such that $\hat{\mathbf{n}} \cdot \mathbf{P} \leq 0$.

---

## Borders and Invalid Depth

The output map is initialized to **`NaN`**. Only interior pixels with valid central differences receive a computed normal.

| Region | Normal value | Reason |
|--------|--------------|--------|
| Border ($r$ px, default 2) | `NaN` | No full $u \pm r$ / $v \pm r$ neighborhood |
| Interior, $\|\mathbf{n}\| < \tau$ | `NaN` | Degenerate cross product before normalization |
| Interior, invalid depth | `NaN` | Non-finite or $d \leq \epsilon$ |
| Interior, valid | Unit vector | Smoothed depth + FD + normalize + face camera |

**Training:** build a mask with `torch.isfinite(normal_map).all(dim=0)` (or per-channel) so losses and gradients ignore `NaN` pixels.

**Debug visualization:** `debug_maps.py` maps `NaN` → 0 for PNG output via `torch.nan_to_num(..., nan=0.0)` before the $(\hat{\mathbf{n}} + 1)/2$ color mapping. Raw `normals.npy` keeps `NaN`.

---

## Coordinate Frame

Normals are in the **same camera coordinate system** as COLMAP / training cameras:

- $+X$ right, $+Y$ down, $+Z$ forward
- Consistent with depth $z = P_z$ used in alignment

To use normals in **world space**, apply the camera rotation (e.g. with $R_{\text{colmap}}$ from the view pose). That transform is not applied in the current loading path.

---

## Computation

- Implemented in **PyTorch** (fully vectorized over the image grid).
- Run **once per camera** at load time in `loadCam`, then stored on `Camera.normal_map` on the training device.
- Training loop exposes `viewpoint_cam.normal_map` as `(3, H, W)`.

Normals are computed **after** COLMAP scale alignment so geometry matches metric depth.

---

## Visualization

For inspection (`debug_maps.py`):

- Raw array: `normals.npy`, shape `(3, H, W)` (includes `NaN` on borders)
- RGB preview: `nan_to_num` → 0 on borders, then $\text{RGB} = 0.5\,(\hat{\mathbf{n}} + 1)$ clamped to $[0,1]$

---

## Relation to `utils/point_utils.py`

`point_utils.depth_to_normal` backprojects via the full projection / world-view pipeline and produces world-oriented normals for **rendered** depth in the rasterizer.

The pipeline described here is specific to **DA3 / frame_data depth**: pinhole backprojection in camera space, used for ground-truth normals from predicted depth maps.
