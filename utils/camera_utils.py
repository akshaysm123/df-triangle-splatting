#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.DA3_utils import prepare_map_for_camera
from utils.depth_colmap_align import align_depth_to_colmap, load_colmap_sparse
from utils.depth_to_normal import depth_map_to_normals

WARNED = False
_colmap_cache = {}

def loadCam(args, id, cam_info, resolution_scale, colmap_data=None):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    orig_size = (orig_w, orig_h)
    train_w, train_h = resolution

    depth_np = prepare_map_for_camera(cam_info.depth, orig_size, resolution)
    if depth_np is not None:
        depth_hw, depth_scale = align_depth_to_colmap(
            depth_np[0].numpy(),
            cam_info,
            colmap_data,
            train_w,
            train_h,
        )
        depth_map = torch.from_numpy(depth_hw[np.newaxis, ...].astype(np.float32))
    else:
        depth_map = None
        depth_scale = 1.0

    confidence_map = prepare_map_for_camera(cam_info.confidence, orig_size, resolution)

    normal_map = None
    if depth_map is not None:
        fx = fov2focal(cam_info.FovX, train_w)
        fy = fov2focal(cam_info.FovY, train_h)
        normal_map = depth_map_to_normals(
            depth_map,
            fx=fx,
            fy=fy,
            cx=train_w / 2.0,
            cy=train_h / 2.0,
        )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  depth_map=depth_map, confidence_map=confidence_map, normal_map=normal_map,
                  depth_scale=depth_scale)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    source_path = getattr(args, "source_path", None)
    colmap_data = None
    if source_path is not None:
        if source_path not in _colmap_cache:
            _colmap_cache[source_path] = load_colmap_sparse(source_path)
        colmap_data = _colmap_cache[source_path]
        if colmap_data is not None:
            print("[ INFO ] Aligning depth maps to COLMAP scale per view")

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, colmap_data))

    if colmap_data is not None and camera_list:
        scales = [cam.depth_scale for cam in camera_list if cam.depth_map is not None]
        if scales:
            print(
                f"[ INFO ] Depth COLMAP scale: min={min(scales):.6f}, "
                f"max={max(scales):.6f}, mean={np.mean(scales):.6f}"
            )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry