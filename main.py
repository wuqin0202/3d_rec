#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与 vggt_feature_m2 对齐的主入口：

- 读取 VGGT 推理结果（相机内外参、深度与置信度）
- 使用 CLIP_Surgery 提取 patch 级视觉特征（忽略 CLS）
- 按 vggt_feature_m2 的坐标/网格/权重融合逻辑构建 3D 体素特征图
- 保存体素图（特征、位置、权重、占用与 RGB）

代码风格：
- 字符串使用单引号
- 使用 logging 进行日志管理
- 使用 Google 风格 Docstring
"""

from __future__ import annotations

import os
import argparse
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np

from vggt_results_saver import (
    load_vggt_results,
    get_image_info_by_index,
)

# 延后导入与可选依赖
import torch
from PIL import Image
import cv2
import clip  # 本仓库的 CLIP_Surgery 扩展模块
from vggt.utils.load_fn import load_and_preprocess_images
from vlmaps.utils.mapping_utils import get_sim_cam_mat, project_point, depth2pc
from vlmaps.utils.mapping_utils import transform_pc, base_pos2grid_id_3d, save_3d_map

# -----------------------------------------------------------------------------
# logging 配置
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 数据访问与适配（VGGT -> 通用接口）
# -----------------------------------------------------------------------------
class VGGTResultReader:
    """VGGT 推理结果读取器。

    将 HDF5 文件中的相机参数与深度图按帧提供出来。

    Note:
        - saver 中保存的 extrinsics_4x4 通常为世界到相机的变换（world->camera）。
          为了与后续模块统一，这里返回相机到世界（camera->world）的 4x4 矩阵。

    Attributes:
        results_path: HDF5 文件路径
        results: 解析后的内存结构
    """

    def __init__(self, results_path: str) -> None:
        """初始化并加载结果文件。

        Args:
            results_path: 由 vggt_results_saver.py 生成的 HDF5 文件路径
        """
        self.results_path = results_path
        self.results = load_vggt_results(results_path)
        logger.info(
            'Loaded VGGT results: %s | images=%d, size=%dx%d',
            results_path,
            int(self.results['metadata']['num_images']),
            int(self.results['metadata']['image_height']),
            int(self.results['metadata']['image_width']),
        )

    @property
    def num_images(self) -> int:
        """帧数量。"""
        return int(self.results['metadata']['num_images'])

    def get_frame(self, index: int) -> Dict[str, Any]:
        """获取指定帧的所有关键信息。

        Args:
            index: 图像索引（0-based）

        Returns:
            包含以下键的字典：
            - image_path: str
            - intrinsics: np.ndarray (3, 3)
            - cam_to_world: np.ndarray (4, 4)
            - depth: np.ndarray (H, W)
            - depth_conf: np.ndarray (H, W)
        """
        info = get_image_info_by_index(self.results, index)
        extrinsics_4x4 = info['extrinsics_4x4']  # world->camera
        intrinsics = info['intrinsics']
        depth = info['depth_map']
        depth_conf = info['depth_confidence']

        # 将 world->camera 变换转换为 camera->world 变换
        cam_to_world = _invert_se3_4x4(extrinsics_4x4)

        return {
            'image_path': info['image_path'],
            'intrinsics': intrinsics,
            'cam_to_world': cam_to_world,
            'depth': depth,
            'depth_conf': depth_conf,
        }


def _invert_se3_4x4(T_wc: np.ndarray) -> np.ndarray:
    """SE3 4x4 刚体变换求逆。

    Args:
        T_wc: 世界到相机的 4x4 矩阵（world->camera）

    Returns:
        camera->world 的 4x4 矩阵。
    """
    assert T_wc.shape == (4, 4), 'Input must be 4x4 matrix'
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_cw = np.eye(4, dtype=T_wc.dtype)
    T_cw[:3, :3] = R_inv
    T_cw[:3, 3] = t_inv
    return T_cw


# -----------------------------------------------------------------------------
# CLIP_Surgery 特征提取（patch 级，忽略 CLS）
# -----------------------------------------------------------------------------
class ClipSurgeryExtractor:
    """CLIP_Surgery 特征提取器。

    - 使用本仓库的 `clip` 模块加载 CS-ViT-B/16
    - 输出为 patch 级特征，忽略 CLS token

    Note:
        `model.encode_image(image)` 输出形状为 (B, 1+N, C)，其中第 1 个 token 为 CLS。
        这里将其移除并返回 (B, N, C)。若需要 2D 网格，可尝试 sqrt(N) 还原为 (H_p, W_p, C)。
    """

    def __init__(self, model_name='CS-ViT-B/16', device: Optional[str] = None) -> None:
        """初始化模型与预处理。

        Args:
            model_name: 模型名称
            device: 设备标识，'cuda'/'cpu'/None(None 表示自动)
        """
        if device is None or device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model = self.model.eval()
        logger.info('CLIP_Surgery model loaded on %s', self.device)

    @torch.no_grad()
    def extract_patch_features(self, image_path: str) -> Dict[str, Any]:
        """对单张图片提取 patch 级特征。

        Args:
            image_path: 图像路径

        Returns:
            包含以下键：
            - features: torch.Tensor, 形状 (N, C)，已 L2 归一化且去除 CLS
            - grid_hw: Tuple[int, int]，若可平方还原则为 (H_p, W_p)，否则为 (1, N)
            - image_bgr: np.ndarray, 原图 BGR，用于可视化
        """
        pil_img = Image.open(image_path).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 移除 CLS token: (B, 1+N, C) -> (N, C)
        feat = image_features[:, 1:, :].squeeze(0).contiguous()  # (N, C)

        # 估计网格大小（通常为方形网格）
        n_tokens = feat.shape[0]
        g = int(np.sqrt(n_tokens))
        if g * g == n_tokens:
            grid_hw = (g, g)
        else:
            grid_hw = (1, n_tokens)

        return {
            'features': feat,        # (N, C)
            'grid_hw': grid_hw,      # (H_p, W_p) 或 (1, N)
            'image_bgr': image_bgr,  # 原始 BGR 图
        }

    @torch.no_grad()
    def extract_patch_features_from_rgb(self, rgb: np.ndarray) -> Dict[str, Any]:
        """从 RGB 数组提取 patch 特征。

        Args:
            rgb: (H, W, 3) RGB uint8

        Returns:
            同 extract_patch_features
        """
        pil_img = Image.fromarray(rgb)
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        feat = image_features[:, 1:, :].squeeze(0).contiguous()
        n_tokens = feat.shape[0]
        g = int(np.sqrt(n_tokens))
        grid_hw = (g, g) if g * g == n_tokens else (1, n_tokens)
        return {
            'features': feat,
            'grid_hw': grid_hw,
            'image_bgr': image_bgr,
        }


class FeatureExtractorPatch:
    """基于 vggt_feature_m2.py 的 3D 体素特征构建器，但使用 CLIP patch 特征。

    - 与 vggt_feature_m2 对齐：process_frame 接口与网格/坐标/权重融合逻辑
    - 差异：特征来自 patch token，不对 token 进行上采样；通过 get_sim_cam_mat(Hp,Wp)
      将相机坐标点投影至 patch 网格采样
    """

    def __init__(self, grid_size: int, cell_size: float, camera_height: float,
                 conf_percent: float = 25.0, max_points: int = 0, device: str = 'auto') -> None:
        self.device = 'cuda' if (device in ('auto', None) and torch.cuda.is_available()) else ('cpu' if device in ('auto', None) else device)
        self.clip_extractor = ClipSurgeryExtractor(device=self.device)

        self.gs = int(grid_size)
        self.cs = float(cell_size)
        self.vh = int(camera_height / self.cs)
        self.depth_conf_percent = float(conf_percent)
        self.depth_max_points = int(max_points) if max_points is not None else 0

        # 体素地图结构
        self.clip_feat_dim: int = 512  # CLIP ViT-B/16 默认 512
        self._init_feature_map()

        # 高度与可视化
        self.height_map = -100 * np.ones((self.gs, self.gs), dtype=np.float32)
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)

        # 坐标链同 builder
        self.init_base_tf = None
        self.inv_init_base_tf = None
        self.base_transform = np.eye(4, dtype=np.float32)
        self.base2cam_tf = np.eye(4, dtype=np.float32)
        self.frame_count = 0

    def _init_feature_map(self) -> None:
        self.grid_feat = np.zeros((self.gs * self.gs, self.clip_feat_dim), dtype=np.float32)
        self.grid_pos = np.zeros((self.gs * self.gs, 3), dtype=np.int32)
        self.occupied_ids = -1 * np.ones((self.gs, self.gs, self.vh), dtype=np.int32)
        self.weight = np.zeros((self.gs * self.gs), dtype=np.float32)
        self.grid_rgb = np.zeros((self.gs * self.gs, 3), dtype=np.uint8)
        self.max_id = 0

    def _reserve_map_space(self) -> None:
        self.grid_feat = np.concatenate([self.grid_feat, np.zeros_like(self.grid_feat)], axis=0)
        self.grid_pos = np.concatenate([self.grid_pos, np.zeros_like(self.grid_pos)], axis=0)
        self.weight = np.concatenate([self.weight, np.zeros_like(self.weight)], axis=0)
        self.grid_rgb = np.concatenate([self.grid_rgb, np.zeros_like(self.grid_rgb)], axis=0)

    def _out_of_range(self, row: int, col: int, height: int) -> bool:
        return (col >= self.gs or row >= self.gs or height >= self.vh or col < 0 or row < 0 or height < 0)

    def _compute_token_intrinsics(self, camera_intrinsic: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        """将相机像素内参映射到 CLIP token 网格坐标系的内参。

        做法：先把像素内参 K 缩放到 CLIP 输入分辨率 n_px（方形，Resize，无 CenterCrop），
        再按 patch 大小 p 把单位从像素转换为 token 索引（grid 坐标）。

        Args:
            camera_intrinsic: (3, 3) 图像像素坐标系下的内参 K。
            img_h: 当前 `rgb_image` 的高度，与 K 的坐标系一致。
            img_w: 当前 `rgb_image` 的宽度，与 K 的坐标系一致。

        Returns:
            (3, 3) 的 token 坐标系内参矩阵 K_tok。
        """
        visual = self.clip_extractor.model.visual
        # CLIP 输入分辨率（方形），例如 224、336
        n_px = int(getattr(visual, 'input_resolution', 224))
        # ViT 的 patch 大小（conv1 的 stride）
        conv1 = getattr(visual, 'conv1', None)
        if conv1 is None:
            # 对 ResNet 视觉编码器不支持 token 网格；当前仅支持 ViT
            raise ValueError('CLIP visual backbone without conv1 (ViT) is not supported for token intrinsics.')
        stride = conv1.stride
        patch = int(stride[0] if isinstance(stride, tuple) else stride)

        # 从图像像素坐标系缩放到 CLIP 输入像素坐标系
        sx = float(n_px) / float(img_w)
        sy = float(n_px) / float(img_h)
        S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        K_clip = (S @ camera_intrinsic).astype(np.float32)

        # 从像素单位转到 token 单位（每个 token 覆盖 patch×patch 像素）
        K_tok = K_clip.copy()
        K_tok[0, :] /= float(patch)
        K_tok[1, :] /= float(patch)

        # 可选：数值稳定与主点范围的轻度裁剪
        return K_tok

    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray, depth_conf: np.ndarray,
                      camera_intrinsic: np.ndarray, camera_pose: np.ndarray) -> Dict[str, Any]:
        """处理单帧（与 vggt_feature_m2 对齐），提取 CLIP patch 特征并更新体素地图。

        Args:
            rgb_image: (H, W, 3) RGB
            depth_image: (H, W) 或 (H, W, 1)
            depth_conf: (H, W)
            camera_intrinsic: (3, 3)
            camera_pose: (4, 4) 相机到世界
        """
        import time
        t_start = time.time()

        # 步骤1：CLIP特征提取
        t1 = time.time()
        if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
            depth_image = depth_image[..., 0]
        feat_dict = self.clip_extractor.extract_patch_features_from_rgb(rgb_image)
        token_feat = feat_dict['features']
        Hp, Wp = feat_dict['grid_hw']
        pix_feats_intr = self._compute_token_intrinsics(camera_intrinsic, rgb_image.shape[0], rgb_image.shape[1])
        t2 = time.time()
        logger.info('process_frame: CLIP特征提取耗时 %.4f 秒', t2 - t1)

        # 步骤2：深度反投影
        t3 = time.time()
        pc, mask = depth2pc(depth_image, intr_mat=camera_intrinsic, min_depth=0., max_depth=6)
        t4 = time.time()
        logger.info('process_frame: 深度反投影耗时 %.4f 秒', t4 - t3)

        # 步骤3：置信度过滤+下采样
        t5 = time.time()
        conf_flat = depth_conf.reshape(-1).astype(np.float32)
        idx_all = np.arange(conf_flat.shape[0])
        idx_valid = idx_all[mask]
        conf_valid = conf_flat[idx_valid]
        if conf_valid.size > 0:
            threshold_val = np.percentile(conf_valid, self.depth_conf_percent)
        else:
            threshold_val = 0.0
        keep = (conf_valid >= threshold_val) & (conf_valid > 1e-5)
        idx_keep = idx_valid[keep]
        if self.depth_max_points > 0 and idx_keep.shape[0] > self.depth_max_points:
            idx_keep = np.random.choice(idx_keep, size=self.depth_max_points, replace=False)
        pc = pc[:, idx_keep]
        t6 = time.time()
        logger.info('process_frame: 置信度过滤+下采样耗时 %.4f 秒', t6 - t5)

        # 步骤4：坐标系对齐
        t7 = time.time()
        if self.init_base_tf is None:
            self.init_base_tf = camera_pose.copy()
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        base_pose = camera_pose
        tf = self.inv_init_base_tf @ base_pose
        pc_transform = tf @ self.base_transform @ self.base2cam_tf
        pc_global = transform_pc(pc, pc_transform)
        t8 = time.time()
        logger.info('process_frame: 坐标系对齐耗时 %.4f 秒', t8 - t7)

        # 步骤5：特征融合
        t9 = time.time()
        processed_points = 0
        C = int(token_feat.shape[1])
        feat_hw_c = token_feat.detach().cpu().to(torch.float32).reshape(Hp, Wp, C).numpy()
        feat_hw_c /= (np.linalg.norm(feat_hw_c, axis=2, keepdims=True) + 1e-6)
        for p_global, p_local in zip(pc_global.T, pc.T):
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p_global[0], p_global[1], p_global[2])
            if self._out_of_range(row, col, height):
                continue
            px, py, pz = project_point(camera_intrinsic, p_local)
            if px < 0 or py < 0 or px >= rgb_image.shape[1] or py >= rgb_image.shape[0]:
                continue
            rgb_v = rgb_image[py, px, :]
            px_feat, py_feat, _ = project_point(pix_feats_intr, p_local)
            if height > self.height_map[row, col]:
                self.height_map[row, col] = height
                self.cv_map[row, col, :] = rgb_v
            if self.max_id >= self.grid_feat.shape[0]:
                self._reserve_map_space()
            radial_dist_sq = float(np.dot(p_local, p_local))
            alpha = float(np.exp(-radial_dist_sq / (2.0 * 0.6)))
            if not (px_feat < 0 or py_feat < 0 or px_feat >= Wp or py_feat >= Hp):
                occupied_id = self.occupied_ids[row, col, height]
                feat_np = feat_hw_c[py_feat, px_feat, :].reshape(-1)
                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id
                    self.grid_feat[self.max_id] = feat_np * alpha
                    self.grid_rgb[self.max_id] = rgb_v
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.max_id += 1
                else:
                    w = self.weight[occupied_id]
                    self.grid_feat[occupied_id] = (self.grid_feat[occupied_id] * w + feat_np * alpha) / (w + alpha)
                    self.grid_rgb[occupied_id] = (self.grid_rgb[occupied_id] * w + rgb_v * alpha) / (w + alpha)
                    self.weight[occupied_id] = w + alpha
                processed_points += 1
        t10 = time.time()
        logger.info('process_frame: 特征融合耗时 %.4f 秒', t10 - t9)

        self.frame_count += 1
        t_end = time.time()
        logger.info('process_frame: 帧 %d 总耗时 %.4f 秒, 点数=%d, 特征数=%d',
                    self.frame_count, t_end - t_start, processed_points, self.max_id)
        return {
            'processed_points': processed_points,
            'total_features': self.max_id,
            'pointcloud_size': pc.shape[1],
            'frame_count': self.frame_count,
            'is_first_frame': self.frame_count == 1,
        }

    def save_feature_map(self, output_path: str) -> None:
        valid_grid_feat = self.grid_feat[: self.max_id]
        valid_grid_pos = self.grid_pos[: self.max_id]
        valid_weight = self.weight[: self.max_id]
        valid_grid_rgb = self.grid_rgb[: self.max_id]
        save_3d_map(
            output_path,
            valid_grid_feat,
            valid_grid_pos,
            valid_weight,
            self.occupied_ids,
            list(range(self.max_id)),
            valid_grid_rgb,
        )
        cv2.imwrite(output_path + '_height_map.png', (self.height_map / self.vh * 255).astype(np.uint8))
        cv2.imwrite(output_path + '_cv_map.png', cv2.cvtColor(self.cv_map, cv2.COLOR_RGB2BGR))
        logger.info('Saved 3D voxel feature map: %s | N=%d', output_path, self.max_id)

def parse_args() -> argparse.Namespace:
    """解析命令行参数（仅保留对齐运行相关）。"""
    parser = argparse.ArgumentParser(
        description='VLMaps-aligned CLIP patch 3D feature mapping',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='CS-ViT-B/16',
        help='Name of the CLIP model to use'
    )
    parser.add_argument(
        '--results',
        type=str,
        default=os.path.join('output', 'vggt_kitchen_results.h5'),
        help='Path to VGGT results .h5 file',
    )
    parser.add_argument(
        '--vlmaps_out',
        type=str,
        default='output/vlmaps_features_clip_patch.h5df',
        help='Output path for the saved 3D voxel feature map',
    )
    parser.add_argument(
        '--vlmaps_grid',
        type=int,
        default=1000,
        help='Grid size for voxel map',
    )
    parser.add_argument(
        '--vlmaps_cell',
        type=float,
        default=0.05,
        help='Cell size (m) for voxel map',
    )
    parser.add_argument(
        '--vlmaps_cam_h',
        type=float,
        default=3.0,
        help='Camera height (m)',
    )
    parser.add_argument(
        '--vlmaps_conf_percent',
        type=float,
        default=25.0,
        help='Depth confidence percentile for filtering',
    )
    parser.add_argument(
        '--vlmaps_max_pts',
        type=int,
        default=0,
        help='Max points per frame after filtering (0 means no subsample)',
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口（仅对齐运行）。"""
    args = parse_args()

    if not os.path.exists(args.results):
        logger.error('Results file not found: %s', args.results)
        return

    reader = VGGTResultReader(args.results)
    fe = FeatureExtractorPatch(
        grid_size=args.vlmaps_grid,
        cell_size=args.vlmaps_cell,
        camera_height=args.vlmaps_cam_h,
        conf_percent=args.vlmaps_conf_percent,
        max_points=args.vlmaps_max_pts,
        device='auto',
    )
    n = reader.num_images
    logger.info('VLMaps-like run over %d frames...', n)
    import time
    for i in range(n):
        frm = reader.get_frame(i)
        # 读取 RGB（按照 metadata 路径）
        img_tensor = load_and_preprocess_images([frm['image_path']])
        rgb = img_tensor[0].permute(1, 2, 0).numpy() * 255.0
        rgb = rgb.astype(np.uint8)

        stats = fe.process_frame(
            rgb_image=rgb,
            depth_image=frm['depth'],
            depth_conf=frm['depth_conf'],
            camera_intrinsic=frm['intrinsics'],
            camera_pose=frm['cam_to_world'],
        )
        if (i + 1) % 5 == 0 or i == n - 1:
            logger.info('Frame %d/%d: points=%d, total_features=%d', i + 1, n, stats['processed_points'], stats['total_features'])

    fe.save_feature_map(args.vlmaps_out)


if __name__ == '__main__':
    main()
