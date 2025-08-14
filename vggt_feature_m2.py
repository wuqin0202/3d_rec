#!/usr/bin/env python3
"""
独立的特征提取脚本
基于VLMapBuilder的特征计算和处理逻辑，提取图像的LSeg特征并构建3D特征地图
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import gdown

# 添加vlmaps到系统路径
sys.path.append(os.getcwd())

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
    save_3d_map,
    load_3d_map,
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet


class FeatureExtractor:
    """特征提取器类，基于VLMapBuilder的逻辑"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self._get_device()

        # 初始化LSeg模型
        self.lseg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()

        # 地图参数
        self.gs = config.grid_size  # 网格大小
        self.cs = config.cell_size  # 单元格大小
        self.vh = int(config.camera_height / self.cs)  # 垂直高度网格数
        self.depth_sample_rate = config.depth_sample_rate

        # 初始化3D特征地图
        self._init_feature_map()

        # 初始化高度地图和可视化地图 (与VLMapBuilder保持一致)
        self.height_map = -100 * np.ones((self.gs, self.gs), dtype=np.float32)
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)

        # 初始化坐标系变换 (与VLMapBuilder保持一致)
        # 使用与VLMapBuilder相同的变量命名：init_base_tf / inv_init_base_tf
        # 并显式保持 base_transform 与 base2cam_tf（此处默认为单位阵，便于与builder逻辑一致）
        self.init_base_tf = None
        self.inv_init_base_tf = None
        self.base_transform = np.eye(4, dtype=np.float32)
        self.base2cam_tf = np.eye(4, dtype=np.float32)
        self.frame_count = 0  # 帧计数器

    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _init_lseg(self) -> Tuple:
        """初始化LSeg模型"""
        crop_size = 480
        base_size = 520

        # 创建LSeg模型
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()

        # 下载和加载预训练权重
        # checkpoint_path = "/data3/qq/proj2/3d_rec/vggt/embedding/lsg_model/demo_e200.ckpt"

        # pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        # pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        # model_state_dict.update(pretrained_state_dict)
        # lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        # 图像预处理参数
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.clip_feat_dim = lseg_model.out_c  # LSeg特征维度，通常是512
        print(f"LSeg特征维度: {self.clip_feat_dim}")

        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _init_feature_map(self):
        """初始化3D特征地图数据结构"""
        # 初始化特征地图相关变量
        # grid_feat: 存储每个占用体素的特征向量 (N, feat_dim)
        self.grid_feat = np.zeros((self.gs * self.gs, self.clip_feat_dim), dtype=np.float32)
        # grid_pos: 存储每个特征对应的网格位置 (N, 3) -> [row, col, height]
        self.grid_pos = np.zeros((self.gs * self.gs, 3), dtype=np.int32)
        # occupied_ids: 3D网格中每个体素对应的特征ID，-1表示未占用 (gs, gs, vh)
        self.occupied_ids = -1 * np.ones((self.gs, self.gs, self.vh), dtype=np.int32)
        # weight: 每个特征的权重，用于特征融合 (N,)
        self.weight = np.zeros((self.gs * self.gs), dtype=np.float32)
        # grid_rgb: 存储每个体素的RGB颜色 (N, 3)
        self.grid_rgb = np.zeros((self.gs * self.gs, 3), dtype=np.uint8)
        # 当前最大特征ID
        self.max_id = 0

        print(f"初始化3D特征地图:")
        print(f"  - 网格大小: {self.gs}x{self.gs}x{self.vh}")
        print(f"  - 单元格大小: {self.cs}m")
        print(f"  - 特征维度: {self.clip_feat_dim}")

    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                     camera_intrinsic: np.ndarray, camera_pose: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像，提取特征并更新3D地图
        与VLMapBuilder保持一致：建立以第一帧为原点的局部世界坐标系

        Args:
            rgb_image: RGB图像 (H, W, 3)
            depth_image: 深度图像 (H, W)
            camera_intrinsic: 相机内参矩阵 (3, 3)
            camera_pose: 相机位姿变换矩阵 (4, 4) - 从相机坐标系到绝对世界坐标系

        Returns:
            包含处理统计信息的字典
        """
        # 1. 提取LSeg特征
        # 输入: rgb_image (H, W, 3)
        # 输出: pix_feats (1, feat_dim, H', W') - 像素级特征图
        pix_feats = get_lseg_feat(
            self.lseg_model, rgb_image, ["example"], self.lseg_transform,
            self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std
        )

        print(f"LSeg特征图尺寸: {pix_feats.shape}")  # 通常是 (1, 512, H', W')

        # 2. 获取特征图对应的相机内参
        # pix_feats的分辨率可能与原图不同，需要相应调整内参
        pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        # 3. 从深度图反投影得到点云
        # 输入: depth_image (H, W), camera_intrinsic (3, 3)
        # 输出: pc (3, N) - 相机坐标系下的点云
        pc, mask = depth2pc(depth_image, intr_mat=camera_intrinsic,
                           min_depth=0.1, max_depth=6)

        # 采样点云以减少计算量
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::self.depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]  # 最终点云 (3, N_sampled)

        print(f"采样后点云数量: {pc.shape[1]}")

        # 4. 建立与VLMapBuilder一致的坐标系链
        # builder中：tf = inv_init_base_tf @ base_pose
        #           pc_transform = tf @ base_transform @ base2cam_tf
        # 这里将 camera_pose 复用为 base_pose，base_transform 与 base2cam_tf 取单位阵，
        # 使得整体链路与builder一致，同时与当前输入保持兼容
        if self.init_base_tf is None:
            self.init_base_tf = camera_pose.copy()
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
            print(f"建立局部坐标系，原点位姿:\n{self.init_base_tf}")

        base_pose = camera_pose
        tf = self.inv_init_base_tf @ base_pose
        pc_transform = tf @ self.base_transform @ self.base2cam_tf

        # 5. 将点云变换到局部世界坐标系（与VLMapBuilder保持一致）
        pc_global = transform_pc(pc, pc_transform)

        # 6. 处理每个3D点，更新特征地图
        processed_points = 0
        for i, (p_global, p_local) in enumerate(zip(pc_global.T, pc.T)):
            # 将世界坐标转换为网格索引
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p_global[0], p_global[1], p_global[2])

            # 检查是否在网格范围内
            if self._out_of_range(row, col, height):
                continue

            # 将3D点投影到RGB图像上获取颜色
            px, py, pz = project_point(camera_intrinsic, p_local)
            if px < 0 or py < 0 or px >= rgb_image.shape[1] or py >= rgb_image.shape[0]:
                continue
            rgb_v = rgb_image[py, px, :]

            # 将3D点投影到特征图上获取特征
            px_feat, py_feat, pz_feat = project_point(pix_feats_intr, p_local)

            # 更新高度地图和可视化地图 (与VLMapBuilder保持一致)
            if height > self.height_map[row, col]:
                self.height_map[row, col] = height
                self.cv_map[row, col, :] = rgb_v

            # 检查是否需要扩展存储空间 (应该在特征更新之前)
            if self.max_id >= self.grid_feat.shape[0]:
                self._reserve_map_space()

            # 应用距离权重 (与VLMapBuilder保持一致)
            # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            # 更新特征地图 (只在特征投影有效时更新)
            if not (px_feat < 0 or py_feat < 0 or px_feat >= pix_feats.shape[3] or py_feat >= pix_feats.shape[2]):
                occupied_id = self.occupied_ids[row, col, height]
                feat = pix_feats[0, :, py_feat, px_feat]
                feat_np = feat.flatten()
                if occupied_id == -1:
                    # 新的占用体素
                    self.occupied_ids[row, col, height] = self.max_id
                    self.grid_feat[self.max_id] = feat_np * alpha
                    self.grid_rgb[self.max_id] = rgb_v
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.max_id += 1
                else:
                    # 已存在的体素，融合特征
                    self.grid_feat[occupied_id] = (
                        self.grid_feat[occupied_id] * self.weight[occupied_id] + feat_np * alpha
                    ) / (self.weight[occupied_id] + alpha)
                    self.grid_rgb[occupied_id] = (
                        self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb_v * alpha
                    ) / (self.weight[occupied_id] + alpha)
                    self.weight[occupied_id] += alpha

                processed_points += 1

        # 更新帧计数器
        self.frame_count += 1

        return {
            'processed_points': processed_points,
            'total_features': self.max_id,
            'feature_shape': pix_feats.shape,
            'pointcloud_size': pc.shape[1],
            'frame_count': self.frame_count,
            'is_first_frame': self.frame_count == 1
        }

    def _out_of_range(self, row: int, col: int, height: int) -> bool:
        """检查网格索引是否超出范围"""
        return (col >= self.gs or row >= self.gs or height >= self.vh or
                col < 0 or row < 0 or height < 0)

    def _reserve_map_space(self):
        """扩展特征地图存储空间 - 与VLMapBuilder保持一致的逻辑"""
        # 使用concatenate方式扩展，与原版保持一致
        self.grid_feat = np.concatenate([
            self.grid_feat,
            np.zeros((self.grid_feat.shape[0], self.grid_feat.shape[1]), dtype=np.float32),
        ], axis=0)

        self.grid_pos = np.concatenate([
            self.grid_pos,
            np.zeros((self.grid_pos.shape[0], self.grid_pos.shape[1]), dtype=np.int32),
        ], axis=0)

        self.weight = np.concatenate([
            self.weight,
            np.zeros((self.weight.shape[0]), dtype=np.float32)  # 修复：使用float32而不是int32
        ], axis=0)

        self.grid_rgb = np.concatenate([
            self.grid_rgb,
            np.zeros((self.grid_rgb.shape[0], self.grid_rgb.shape[1]), dtype=np.uint8),  # 修复：使用uint8
        ], axis=0)

        print(f"扩展存储空间: {self.grid_feat.shape[0] // 2} -> {self.grid_feat.shape[0]}")

    def save_feature_map(self, output_path: Path):
        """保存特征地图"""
        # 只保存有效的特征
        valid_grid_feat = self.grid_feat[:self.max_id]
        valid_grid_pos = self.grid_pos[:self.max_id]
        valid_weight = self.weight[:self.max_id]
        valid_grid_rgb = self.grid_rgb[:self.max_id]

        save_3d_map(
            output_path,
            valid_grid_feat,
            valid_grid_pos,
            valid_weight,
            self.occupied_ids,
            list(range(self.max_id)),  # 映射的帧列表
            valid_grid_rgb
        )
        print(f"特征地图已保存到: {output_path}")
        print(f"总特征数量: {self.max_id}")


def dummy_get_camera_params(image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟函数：获取相机参数
    在实际使用中，这里应该替换为真实的相机参数获取逻辑

    注意：camera_pose应该是绝对世界坐标系下的位姿，
    FeatureExtractor会自动建立以第一帧为原点的局部坐标系

    Returns:
        camera_intrinsic: 相机内参矩阵 (3, 3)
        camera_pose: 相机位姿变换矩阵 (4, 4) - 绝对世界坐标系
        depth: 深度图像 (H, W)
    """
    # 模拟相机内参 (假设640x480图像)
    camera_intrinsic = np.array([
        [540.0, 0.0, 320.0],
        [0.0, 540.0, 240.0],
        [0.0, 0.0, 1.0]
    ])

    # 模拟相机位姿 (绝对世界坐标系，会自动转换为局部坐标系)
    camera_pose = np.eye(4)
    # 假设相机在z=1.5米高度
    camera_pose[2, 3] = 1.5

    # 模拟深度图 (随机深度值，实际应该从传感器获取)
    depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)

    return camera_intrinsic, camera_pose, depth


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="独立的特征提取脚本")
    parser.add_argument("--input_dir", type=str, default= "/data25/wuqin/projects/3d_rec1/vggt/examples/kitchen_simple/images",
                       help="输入图像目录路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录路径")
    parser.add_argument("--config", type=str,
                       default="config/map_config/vlmaps.yaml",
                       help="配置文件路径")
    parser.add_argument("--grid_size", type=int, default=1000,
                       help="网格大小")
    parser.add_argument("--cell_size", type=float, default=0.05,
                       help="单元格大小(米)")
    parser.add_argument("--camera_height", type=float, default=3.0,
                       help="相机高度(米)")
    parser.add_argument("--depth_sample_rate", type=int, default=1,
                       help="深度采样率")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
        # 用命令行参数覆盖配置
        config.grid_size = args.grid_size
        config.cell_size = args.cell_size
        config.camera_height = args.camera_height
        config.depth_sample_rate = args.depth_sample_rate
    else:
        # 创建默认配置
        config = OmegaConf.create({
            'grid_size': args.grid_size,
            'cell_size': args.cell_size,
            'camera_height': args.camera_height,
            'depth_sample_rate': args.depth_sample_rate
        })

    # 初始化特征提取器
    print("初始化特征提取器...")
    extractor = FeatureExtractor(config)

    # 获取输入图像列表
    input_dir = Path(args.input_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(input_dir / ext)))
    image_paths.sort()

    if not image_paths:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_paths)} 张图像")

    # 处理每张图像
    pbar = tqdm(image_paths, desc="处理图像")
    for i, image_path in enumerate(pbar):
        # 加载RGB图像
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"无法加载图像: {image_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 获取相机参数 (这里使用模拟函数，实际应该替换为真实的参数获取)
        camera_intrinsic, camera_pose, depth = dummy_get_camera_params(Path(image_path))

        # 处理帧
        stats = extractor.process_frame(rgb, depth, camera_intrinsic, camera_pose)

        # 更新进度条信息
        pbar.set_postfix({
            'points': stats['processed_points'],
            'features': stats['total_features']
        })

    # 保存最终特征地图
    final_output = output_dir / "vlmaps_features.h5df"
    extractor.save_feature_map(final_output)

    print("特征提取完成!")
    print(f"最终特征地图保存在: {final_output}")
    print(f"总特征数量: {extractor.max_id}")


if __name__ == "__main__":
    main()
