# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import matplotlib
import matplotlib.pyplot as plt
import trimesh
import pyrender
import open3d as o3d
from pathlib import Path

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from scipy.ndimage import distance_transform_edt

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT.from_pretrained("/data23/projs/resources/VGGT/VGGT-1B")  # another way to load the model


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    torch.cuda.synchronize()
    start_time = time.time()
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    torch.cuda.synchronize()
    print(f"{time.time() - start_time:.2f} seconds to compute world points from depth map.")

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def create_top_down_view(
    predictions,
    conf_thres: float = 10,
    solution: float = 0.001,
    proj_surface: str = 'yz',
    proj_range: tuple = None,
    device: str = None,
    save_path: str = None,
    is_save_ply: bool = False,
    is_transformed: bool = False,
    save_name: str = "top_down_view.jpg"
):
    eps = 1e-8

    print("Createing top-down view...")
    start_time = time.time()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    proj_axes = {'xy': (0, 1), 'yz': (1, 2), 'xz': (0, 2)}.get(proj_surface)
    if proj_axes is None:
        raise ValueError(f"Invalid projection surface: {proj_surface}. Choose from 'xy', 'yz', or 'xz'.")
    z_axis = [i for i in range(3) if i not in proj_axes][0]

    # load data to device
    pred_world_points = torch.from_numpy(predictions["world_points_from_depth"]).half().to(device)
    pred_world_points_conf = torch.from_numpy(
        predictions.get("depth_conf", np.ones_like(predictions["world_points_from_depth"][..., 0]))
    ).float().to(device)
    images = torch.from_numpy(predictions["images"]).half().to(device)

    vertices_3d = pred_world_points.reshape(-1, 3)
    vertices_3d = (vertices_3d / solution).int()

    # create colors_rgb
    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = images.permute(0, 2, 3, 1)
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).to(torch.uint8)

    # create confidence mask
    conf = pred_world_points_conf.reshape(-1)
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = torch.quantile(conf, conf_thres / 100.0)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    # filter vertices and colors by confidence mask
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]
    print(f"{time.time() - start_time:.2f} seconds to filter points by confidence.")
    # print(f"Projection direction value range: min={vertices_3d[:, z_axis].min()}, max={vertices_3d[:, z_axis].max()}")

    # transform z-axis if needed
    if is_transformed:
        vertices_3d[:, z_axis] = -vertices_3d[:, z_axis]
        print(f"{time.time() - start_time:.2f} seconds to transform z-axis.")

    # filter by projection range if provided
    if proj_range:
        z_range_mask = (vertices_3d[:, z_axis] >= proj_range[0]) & (vertices_3d[:, z_axis] <= proj_range[1])
        vertices_3d = vertices_3d[z_range_mask]
        colors_rgb = colors_rgb[z_range_mask]
        print(f"{time.time() - start_time:.2f} seconds to filter points by projection range {proj_range}.")

    # check if there are any points left after filtering
    if is_save_ply and vertices_3d.shape[0] > 0:
        ply_path = os.path.join(save_path, "point_cloud.ply") if save_path else "point_cloud.ply"
        point_cloud_data = trimesh.PointCloud(
            vertices=(vertices_3d * solution).cpu().numpy(), colors=colors_rgb.cpu().numpy()
        )
        point_cloud_data.export(ply_path)
        print(f"{time.time() - start_time:.2f} seconds to save point cloud to {ply_path}.")

    # project vertices to 2D
    xy = vertices_3d[:, proj_axes]
    z = vertices_3d[:, z_axis].float()

    # calculate min/max of xy coordinates
    x_min, y_min = torch.min(xy, dim=0).values
    x_max, y_max = torch.max(xy, dim=0).values
    img_size_x, img_size_y = int((x_max - x_min).item() + 1), int((y_max - y_min).item() + 1)
    print(f"{time.time() - start_time:.2f} seconds to compute min/max of xy coordinates.")
    torch.cuda.synchronize()

    # scale coordinates to image size
    scale_x = (xy[:, 0] - x_min) / (x_max - x_min + eps)
    scale_y = (xy[:, 1] - y_min) / (y_max - y_min + eps)
    px = torch.clamp((scale_x * (img_size_x - 1)).long(), 0, img_size_x - 1)
    py = torch.clamp((scale_y * (img_size_y - 1)).long(), 0, img_size_y - 1)

    flat_indices = py * img_size_x + px
    unique_indices, inverse_indices = torch.unique(flat_indices, return_inverse=True)
    max_z_per_pixel = torch.full((unique_indices.shape[0], ), float('-inf'), dtype=z.dtype, device=device)
    max_z_per_pixel.scatter_reduce_(0, inverse_indices, z, reduce='amax', include_self=False)
    mask_torch = (z == max_z_per_pixel[inverse_indices])
    idx = torch.nonzero(mask_torch, as_tuple=True)[0]

    img_torch = torch.ones((img_size_y, img_size_x, 3), dtype=torch.uint8, device=device) * 255
    img_torch[py[idx], px[idx]] = colors_rgb[idx]
    torch.cuda.synchronize()
    print(f"{time.time() - start_time:.2f} seconds to create top-down view image by projecting {z.shape[0]} points.")

    # inpaint empty pixels using nearest neighbor interpolation
    img_np = img_torch.cpu().numpy()
    mask = (img_np == 255).all(axis=2)  # True 表示空白像素

    # Use scipy's distance transform to fill empty pixels
    not_mask = ~mask
    if not_mask.any():
        _, nearest = distance_transform_edt(mask, return_indices=True)
        img_np_filled = img_np[nearest[0], nearest[1]]
        img_np = img_np_filled
    torch.cuda.synchronize()
    print(f"{time.time() - start_time:.2f} seconds to inpaint top-down view image with torch/scipy (nearest neighbor).")

    # apply Gaussian blur to smooth the image
    img_np = cv2.GaussianBlur(img_np, (5, 5), sigmaX=1.5)
    img_torch_interp = torch.from_numpy(img_np).to(torch.uint8).to(device)
    torch.cuda.synchronize()
    print(
        f"{time.time() - start_time:.2f} seconds to create top-down view image by processing {z.shape[0]} points (with fast inpaint interpolation and Gaussian blur)."
    )

    save_path = os.path.join(save_path, save_name) if save_path or save_name else None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, img_torch_interp.cpu().numpy())
        print(f"{time.time() - start_time:.2f} seconds to create top-down view image.")
        print(f"Top-down view image saved to: {os.path.abspath(save_path)}")
    print(f"Total time for creating top-down view: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    conf_thres = 25
    solution = 0.001
    target_dir = 'examples/hall_4f_25'

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    create_top_down_view(
        predictions=predictions,
        conf_thres=conf_thres,
        solution=solution,
        proj_surface='xz',
        proj_range=(-185, 35),
        device=device,
        is_save_ply=False,
        is_transformed=True,
        save_path='./',
        save_name="top_down_view.jpg"
    )

