#
# Render per-pixel transmittance depth ranges and export one NPZ per COLMAP image.
#

import os
from typing import Optional

import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.general_utils import safe_state


def render_depth_ranges(model_path, name, iteration, views, triangles, pipeline, background, output_dir, quick):
    os.makedirs(output_dir, exist_ok=True)

    if quick:
        views = views[:10]

    for view in tqdm(views, desc=f"Depth ranges ({name})"):
        pkg = render(view, triangles, pipeline, background)
        depth_t05 = pkg["depth_transmittance_05"][0].detach().float().cpu().numpy()
        depth_t95 = pkg["depth_transmittance_95"][0].detach().float().cpu().numpy()
        alpha = pkg["rend_alpha"][0].detach().float().cpu().numpy()

        npz_path = os.path.join(output_dir, f"{view.image_name}.npz")
        np.savez(
            npz_path,
            depth_t05=depth_t05,
            depth_t95=depth_t95,
            alpha=alpha,
        )


def main(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, quick: bool, output_dir: Optional[str]):
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            init_size=None,
            nb_points=None,
            set_sigma=None,
            no_dome=False,
            load_iteration=iteration,
            shuffle=False,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        base_output = output_dir or os.path.join(
            dataset.model_path,
            "depth_ranges",
            f"ours_{scene.loaded_iter}",
        )

        if not skip_train:
            train_out = os.path.join(base_output, "train")
            render_depth_ranges(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                triangles,
                pipeline,
                background,
                train_out,
                quick,
            )

        if not skip_test:
            test_out = os.path.join(base_output, "test")
            render_depth_ranges(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                triangles,
                pipeline,
                background,
                test_out,
                quick,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Export transmittance depth ranges as NPZ files")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_dir", type=str, default=None, help="Root output directory for NPZ files")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Only process the first 10 views")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering depth ranges for " + args.model_path)

    safe_state(args.quiet)
    main(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.quick,
        getattr(args, "output_dir", None),
    )
