#
# Debug script: load RGB, depth, and confidence the same way as train.py and save
# visualizations for manual alignment checks.

import os
import sys
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser
from matplotlib import cm
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from scene import Scene, TriangleModel
from utils.general_utils import safe_state


def scalar_to_colormap(tensor_1hw, cmap_name="turbo", valid_mask=None):
    """Map a single-channel tensor [1, H, W] to RGB [3, H, W] for inspection."""
    x = tensor_1hw[0].detach().float().cpu()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    if valid_mask.any():
        vals = x[valid_mask]
        lo = torch.quantile(vals, 0.02)
        hi = torch.quantile(vals, 0.98)
    else:
        lo, hi = x.min(), x.max()
    if hi <= lo:
        hi = lo + 1e-6
    x_norm = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(x_norm.numpy())
    rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).float()
    rgb[:, ~valid_mask] = 0.0
    return rgb


def save_debug_views(views, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for view in tqdm(views, desc="Saving debug maps"):
        name = view.image_name
        view_dir = os.path.join(output_dir, name)
        os.makedirs(view_dir, exist_ok=True)

        rgb = view.original_image[:3].detach().cpu()
        torchvision.utils.save_image(rgb, os.path.join(view_dir, "rgb.png"))

        if view.depth_map is None or view.confidence_map is None:
            raise RuntimeError(
                f"Camera '{name}' is missing depth_map or confidence_map. "
                "Check that frame_data/ exists and maps were loaded."
            )

        depth = view.depth_map.detach().cpu()
        confidence = view.confidence_map.detach().cpu()

        np.save(os.path.join(view_dir, "depth.npy"), depth.numpy())
        np.save(os.path.join(view_dir, "confidence.npy"), confidence.numpy())

        depth_vis = scalar_to_colormap(depth, valid_mask=torch.isfinite(depth[0]))
        conf_vis = scalar_to_colormap(confidence, cmap_name="viridis", valid_mask=torch.isfinite(confidence[0]))
        torchvision.utils.save_image(depth_vis, os.path.join(view_dir, "depth_vis.png"))
        torchvision.utils.save_image(conf_vis, os.path.join(view_dir, "confidence_vis.png"))

        composite = torch.cat([rgb, depth_vis, conf_vis], dim=2)
        torchvision.utils.save_image(composite, os.path.join(view_dir, "rgb_depth_confidence.png"))


def main(dataset, opt, no_dome, debug_dir, skip_train, skip_test, max_views):
    if not dataset.model_path:
        dataset.model_path = os.path.join("./output", "debug_maps_run")
    os.makedirs(dataset.model_path, exist_ok=True)

    if debug_dir is None:
        debug_dir = os.path.join(dataset.model_path, "debug_maps")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Saving debug outputs to {debug_dir}")

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        triangles,
        opt.set_opacity,
        opt.triangle_size,
        opt.nb_points,
        opt.set_sigma,
        no_dome,
        shuffle=False,
    )

    views = []
    if not skip_train:
        views.extend(scene.getTrainCameras())
    if not skip_test:
        views.extend(scene.getTestCameras())

    if max_views > 0:
        views = views[:max_views]

    print(f"Exporting {len(views)} views")
    save_debug_views(views, debug_dir)
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export RGB / depth / confidence for alignment checks")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--debug_dir", type=str, default=None, help="Output folder (default: <model_path>/debug_maps)")
    parser.add_argument("--max_views", type=int, default=-1, help="Limit number of views (-1 = all)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    main(
        lp.extract(args),
        op.extract(args),
        args.no_dome,
        args.debug_dir,
        args.skip_train,
        args.skip_test,
        args.max_views,
    )
