#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLIP_Surgery语言引导3D高亮功能实现

基于CLIP_Surgery模型的3D点云高亮查询系统。
相比原版demo_clip_highlight.py，此版本使用CLIP_Surgery直接提取patch级特征，
避免了网格切分的多次推理过程。

主要功能：
1. 加载VGGT重建结果
2. 使用CLIP_Surgery预计算图像patch特征
3. 启动Viser服务器并集成高亮功能
4. 提供交互式语言查询界面

使用方法：
python main.py --data_dir <重建数据目录>

作者：VGGT Team
"""

import os
import sys
import argparse
import time
import threading
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image
import gc

# 导入必要模块
import viser
import viser.transforms
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

# 导入CLIP_Surgery
import clip


class ClipSurgeryFeatureManager:
    """CLIP_Surgery特征管理器 - 核心类"""

    def __init__(self, model_name: str = "CS-ViT-B/16"):
        """
        初始化CLIP_Surgery特征管理器

        Args:
            model_name: CLIP_Surgery模型名称
        """
        self.model_name = model_name
        self.similarity_threshold = 0.25
        self.highlight_color = [255, 255, 0]
        self.point_size = 0.003

        # 下采样配置
        self.enable_downsampling = True
        self.max_points = 50000
        self.downsample_method = 'uniform'

        # 初始化模型
        self.device = self._get_device()
        self.clip_model = None
        self.clip_preprocess = None
        self._load_clip_surgery_model()

        # 特征缓存
        self.feature_cache = {}
        self.scene_hash = None

        print(f"✅ ClipSurgeryFeatureManager初始化完成")
        print(f"   模型: {self.model_name}")
        print(f"   设备: {self.device}")

    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if visible_devices is not None:
                device = "cuda:0"
            else:
                device = "cuda"

            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"🚀 CLIP_Surgery使用GPU设备: {device} ({device_name})")
            return device
        else:
            print("⚠️ CUDA不可用，CLIP_Surgery使用CPU")
            return "cpu"

    def _load_clip_surgery_model(self):
        """加载CLIP_Surgery模型"""
        try:
            print(f"🔄 正在加载CLIP_Surgery模型: {self.model_name}")
            start_time = time.time()

            self.clip_model, self.clip_preprocess = clip.load(self.model_name, device=self.device)
            self.clip_model.eval()

            load_time = time.time() - start_time
            print(f"✅ CLIP_Surgery模型加载完成 ({load_time:.2f}s)")

        except Exception as e:
            print(f"❌ CLIP_Surgery模型加载失败: {e}")
            raise

    def _generate_scene_hash(self, predictions: Dict[str, Any]) -> str:
        """生成场景的唯一哈希值"""
        try:
            import hashlib

            # 使用图像数据和相机参数生成哈希
            images = predictions['images']
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']

            # 创建哈希输入
            hash_input = []
            hash_input.append(images.shape)
            hash_input.append(np.mean(images.cpu().numpy() if hasattr(images, 'cpu') else images))
            hash_input.append(extrinsics.flatten())
            hash_input.append(intrinsics.flatten())

            # 生成MD5哈希
            hash_str = str(hash_input).encode('utf-8')
            scene_hash = hashlib.md5(hash_str).hexdigest()[:16]

            return scene_hash

        except Exception as e:
            print(f"⚠️ 场景哈希生成失败: {e}")
            return f"scene_{int(time.time())}"

    def precompute_features(self, predictions: Dict[str, Any]) -> str:
        """
        预计算阶段：使用CLIP_Surgery提取图像patch特征并建立3D-2D索引

        Args:
            predictions: VGGT模型预测结果

        Returns:
            场景哈希值
        """
        print("🚀 开始CLIP_Surgery特征预计算...")
        start_time = time.time()

        # 生成场景哈希
        print("📝 步骤 1/3: 生成场景哈希...")
        scene_hash = self._generate_scene_hash(predictions)
        self.scene_hash = scene_hash
        print(f"🎯 场景哈希: {scene_hash}")

        # 提取图像patch特征
        print("🖼️ 步骤 2/3: 提取CLIP_Surgery patch特征...")
        feature_start_time = time.time()
        image_features = self._extract_clip_surgery_features(predictions['images'])
        feature_time = time.time() - feature_start_time
        print(f"✅ 图像特征提取完成 ({feature_time:.2f}s)")

        # 建立3D-2D映射索引
        print("🗺️ 步骤 3/3: 建立3D-2D patch映射索引...")
        mapping_start_time = time.time()
        point_features = self._build_3d_2d_patch_mapping(predictions, image_features)
        mapping_time = time.time() - mapping_start_time
        print(f"✅ 3D-2D映射完成 ({mapping_time:.2f}s)")

        # 下采样以提高查询性能
        if self.enable_downsampling:
            print("🔽 额外步骤: 下采样3D点以提高性能...")
            downsample_start_time = time.time()
            point_features = self._downsample_point_features(
                point_features,
                max_points=self.max_points,
                method=self.downsample_method
            )
            downsample_time = time.time() - downsample_start_time
            print(f"✅ 下采样完成 ({downsample_time:.2f}s)")

        # 保存特征
        features = {
            'scene_hash': scene_hash,
            'point_features': point_features,
            'image_features': image_features,
            'predictions_meta': {
                'num_views': predictions['images'].shape[0],
                'image_shape': predictions['images'].shape[2:],
                'device': str(predictions['images'].device) if hasattr(predictions['images'], 'device') else 'cpu'
            }
        }

        # 缓存特征
        self.feature_cache = features

        total_time = time.time() - start_time
        num_points = len(point_features)
        print(f"✅ CLIP_Surgery特征预计算完成: {num_points} 个3D点 ({total_time:.2f}s)")

        return scene_hash

    def _extract_clip_surgery_features(self, images) -> Dict[int, Dict[str, any]]:
        """
        使用CLIP_Surgery提取patch级特征

        Args:
            images: 图像数据 (S, 3, H, W)

        Returns:
            每个视图的patch特征字典 {view_idx: {'patch_features': tensor, 'patch_coords': coords}}
        """
        image_features = {}

        # 处理不同的输入类型
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        elif isinstance(images, torch.Tensor):
            images = images.cpu().float()

        S, C, H, W = images.shape
        print(f"   📊 开始CLIP_Surgery处理 {S} 个视图，图像尺寸: {H}x{W}")

        with torch.no_grad():
            for i in range(S):
                print(f"   🔄 处理视图 {i+1}/{S}...")

                img = images[i]  # (3, H, W)

                # 转换为PIL图像并预处理
                if isinstance(img, torch.Tensor):
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    img_np = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

                pil_img = Image.fromarray(img_np)

                # CLIP_Surgery预处理
                processed_img = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

                # 使用CLIP_Surgery提取特征
                # encode_image返回的是 (B, N, D) 其中 N = HW/patch_size^2 + 1 (包含CLS token)
                patch_features = self.clip_model.encode_image(processed_img)  # (1, N, D)

                # 移除batch维度和CLS token，只保留patch特征
                patch_features = patch_features[0, 1:, :]  # (N-1, D) 移除CLS token

                # 计算patch grid的大小
                # 假设input_resolution=224, patch_size=16, 则有 (224/16)^2 = 196 个patches
                patch_size = 16 if "16" in self.model_name else 32 if "32" in self.model_name else 14
                input_resolution = 224  # CLIP默认输入尺寸
                patches_per_side = input_resolution // patch_size

                print(f"     patch数量: {patch_features.shape[0]}, 特征维度: {patch_features.shape[1]}")
                print(f"     patch grid: {patches_per_side}x{patches_per_side}")

                # 生成patch坐标
                patch_coords = []
                patch_h = H / patches_per_side
                patch_w = W / patches_per_side

                for patch_idx in range(patch_features.shape[0]):
                    row = patch_idx // patches_per_side
                    col = patch_idx % patches_per_side

                    # 计算在原图中的坐标范围
                    y_start = int(row * patch_h)
                    y_end = int((row + 1) * patch_h)
                    x_start = int(col * patch_w)
                    x_end = int((col + 1) * patch_w)

                    patch_coords.append({
                        'patch_idx': patch_idx,
                        'row': row,
                        'col': col,
                        'y_start': y_start,
                        'y_end': y_end,
                        'x_start': x_start,
                        'x_end': x_end,
                        'center_y': (y_start + y_end) // 2,
                        'center_x': (x_start + x_end) // 2
                    })

                image_features[i] = {
                    'patch_features': patch_features,  # (N, D)
                    'patch_coords': patch_coords,      # List[Dict]
                    'patch_grid_shape': (patches_per_side, patches_per_side),
                    'original_shape': (H, W)
                }

                # 显示进度
                progress_percent = ((i + 1) / S) * 100
                print(f"   ✅ 视图 {i+1}/{S} 完成 ({progress_percent:.1f}%) - {patch_features.shape[0]} 个patches")

                # 清理GPU内存
                if self.device.startswith('cuda') and (i + 1) % 2 == 0:
                    torch.cuda.empty_cache()

        # 清理GPU内存
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            gc.collect()

        return image_features

    def _build_3d_2d_patch_mapping(self, predictions: Dict[str, Any], image_features: Dict[int, Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        基于CLIP_Surgery patches的3D-2D映射索引建立

        Args:
            predictions: VGGT预测结果
            image_features: patch特征字典

        Returns:
            3D点特征字典 {point_key: patch_feature_tensor}
        """
        # 确保数据格式正确
        depth = predictions["depth"]
        extrinsic = predictions["extrinsic"]
        intrinsic = predictions["intrinsic"]

        # 转换为torch张量（如果是numpy数组）
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        if isinstance(extrinsic, np.ndarray):
            extrinsic = torch.from_numpy(extrinsic).float()
        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic).float()

        # 生成世界坐标点云
        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

        S, H, W, _ = world_points.shape
        point_features = {}

        print(f"   处理 {S} 个视图的patch 3D-2D映射，图像尺寸: {H}x{W}")

        # 计算总的像素数用于进度显示
        total_pixels = S * H * W
        processed_pixels = 0

        for view_idx in range(S):
            if view_idx not in image_features:
                continue

            patch_data = image_features[view_idx]
            patch_features = patch_data['patch_features']  # (N, D)
            patch_coords = patch_data['patch_coords']      # List[Dict]
            patch_grid_shape = patch_data['patch_grid_shape']  # (rows, cols)

            view_world_points = world_points[view_idx]  # (H, W, 3)

            print(f"   视图 {view_idx}: patch grid {patch_grid_shape}, 图像尺寸 {H}x{W}")

            # 遍历每个像素，添加进度显示
            pixels_in_view = H * W
            progress_step = max(1, pixels_in_view // 20)  # 每5%显示一次进度

            for y in range(H):
                for x in range(W):
                    # 获取3D点坐标
                    point_3d = view_world_points[y, x]  # (3,)

                    # 检查点的有效性
                    if np.isnan(point_3d).any() or np.isinf(point_3d).any():
                        processed_pixels += 1
                        continue

                    # 找到对应的patch
                    patch_found = False
                    for patch_coord in patch_coords:
                        if (patch_coord['y_start'] <= y < patch_coord['y_end'] and
                            patch_coord['x_start'] <= x < patch_coord['x_end']):

                            # 获取对应的patch特征
                            patch_idx = patch_coord['patch_idx']
                            patch_feature = patch_features[patch_idx]  # (D,)

                            # 创建3D点的键（量化坐标以减少内存使用）
                            point_key = f"{point_3d[0]:.3f},{point_3d[1]:.3f},{point_3d[2]:.3f}"

                            # 存储点特征
                            point_features[point_key] = patch_feature
                            patch_found = True
                            break

                    if not patch_found:
                        print(f"   ⚠️ 警告: 像素 ({y}, {x}) 未找到对应patch")

                    processed_pixels += 1

                    # 显示进度
                    if processed_pixels % progress_step == 0:
                        progress_percent = (processed_pixels / total_pixels) * 100
                        current_view_progress = ((y * W + x + 1) / pixels_in_view) * 100
                        print(f"     视图 {view_idx}: {current_view_progress:.1f}% | 总进度: {progress_percent:.1f}% | 已建立 {len(point_features)} 个3D点")

            print(f"   ✅ 视图 {view_idx} 完成，当前3D点总数: {len(point_features)}")

        print(f"   建立了 {len(point_features)} 个3D点的patch特征索引")

        return point_features

    def _downsample_point_features(self, point_features: Dict[str, torch.Tensor],
                                 max_points: int = 50000,
                                 method: str = "uniform") -> Dict[str, torch.Tensor]:
        """
        下采样3D点特征以提高查询性能

        Args:
            point_features: 原始3D点特征字典
            max_points: 最大保留点数
            method: 下采样方法 ("uniform", "random")

        Returns:
            下采样后的3D点特征字典
        """
        original_count = len(point_features)

        if original_count <= max_points:
            print(f"   点数 {original_count} 已在限制内，无需下采样")
            return point_features

        print(f"   原始点数: {original_count}, 目标点数: {max_points}, 方法: {method}")

        if method == "uniform":
            # 均匀下采样：每隔固定间隔选择点
            step = original_count // max_points
            selected_keys = list(point_features.keys())[::step][:max_points]

        elif method == "random":
            # 随机下采样
            import random
            all_keys = list(point_features.keys())
            selected_keys = random.sample(all_keys, min(max_points, len(all_keys)))

        else:
            raise ValueError(f"未知的下采样方法: {method}")

        # 创建下采样后的字典
        downsampled_features = {key: point_features[key] for key in selected_keys}

        final_count = len(downsampled_features)
        reduction_ratio = (original_count - final_count) / original_count * 100

        print(f"   下采样完成: {original_count} -> {final_count} 点 (减少 {reduction_ratio:.1f}%)")

        return downsampled_features

    def query_and_highlight(self, text_query: str, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        语言查询阶段：根据文本查询高亮3D点

        Args:
            text_query: 查询文本
            threshold: 相似度阈值，None则使用默认值

        Returns:
            (highlighted_points, highlighted_colors, stats)
        """
        if not self.feature_cache:
            raise ValueError("特征缓存为空，请先调用precompute_features()")

        if threshold is None:
            threshold = self.similarity_threshold

        print(f"🔍 语言查询: '{text_query}' (阈值: {threshold:.3f})")
        start_time = time.time()

        # 编码文本查询
        text_features = self._encode_text_query(text_query)

        # 计算相似度并筛选点
        highlighted_points, highlighted_colors, stats = self._compute_similarity_and_filter(
            text_features, threshold
        )

        query_time = time.time() - start_time
        print(f"✅ 查询完成: {len(highlighted_points)} 个高亮点 ({query_time:.3f}s)")

        return highlighted_points, highlighted_colors, stats

    def _encode_text_query(self, text_query: str) -> torch.Tensor:
        """编码文本查询"""
        with torch.no_grad():
            # 使用CLIP_Surgery的文本编码
            text_features = clip.encode_text_with_prompt_ensemble(
                self.clip_model, [text_query], self.device
            )
            text_features = text_features[0]  # 取第一个（也是唯一一个）文本的特征

            print(f"   📝 文本特征形状: {text_features.shape}")

            return text_features

    def _compute_similarity_and_filter(self, text_features: torch.Tensor, threshold: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """计算相似度并过滤高亮点"""
        point_features = self.feature_cache['point_features']

        highlighted_points = []
        highlighted_scores = []
        total_points = len(point_features)
        processed_points = 0

        print(f"   🔄 计算 {total_points} 个3D点的patch相似度...")

        # 逐点计算相似度
        for point_key, patch_feature in point_features.items():
            # 解析3D坐标
            coords = [float(x) for x in point_key.split(',')]
            point_3d = np.array(coords)

            # 归一化patch特征
            patch_feature_norm = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            similarity = torch.dot(text_features, patch_feature_norm).item()

            # 如果相似度超过阈值，添加到高亮列表
            if similarity >= threshold:
                highlighted_points.append(point_3d)
                highlighted_scores.append(similarity)

            processed_points += 1
            if processed_points % 10000 == 0:
                print(f"   已处理 {processed_points}/{total_points} 个点，当前高亮点数: {len(highlighted_points)}")

        # 转换为numpy数组
        if highlighted_points:
            highlighted_points = np.array(highlighted_points)
            highlighted_scores = np.array(highlighted_scores)

            # 生成颜色（基于相似度）
            highlighted_colors = self._generate_highlight_colors(highlighted_scores)
        else:
            highlighted_points = np.empty((0, 3))
            highlighted_colors = np.empty((0, 3))
            highlighted_scores = np.array([])

        # 统计信息
        stats = {
            'total_points': total_points,
            'highlighted_points': len(highlighted_points),
            'threshold': threshold,
            'max_similarity': float(np.max(highlighted_scores)) if len(highlighted_scores) > 0 else 0.0,
            'min_similarity': float(np.min(highlighted_scores)) if len(highlighted_scores) > 0 else 0.0,
            'mean_similarity': float(np.mean(highlighted_scores)) if len(highlighted_scores) > 0 else 0.0,
            'matching_method': 'clip_surgery_patch'
        }

        return highlighted_points, highlighted_colors, stats

    def _generate_highlight_colors(self, scores: np.ndarray) -> np.ndarray:
        """根据相似度分数生成高亮颜色"""
        if len(scores) == 0:
            return np.empty((0, 3))

        # 归一化分数到[0, 1]
        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores)

        # 生成颜色：从红色到黄色渐变
        colors = []
        base_color = np.array(self.highlight_color)  # 默认黄色

        for score in normalized_scores:
            # 基于分数调整颜色强度
            intensity = 0.5 + 0.5 * score  # 0.5到1.0的强度
            color = base_color * intensity
            colors.append(color.astype(np.uint8))

        return np.array(colors)

    def clear_cache(self):
        """清理内存缓存"""
        self.feature_cache.clear()
        self.scene_hash = None

        # 清理GPU内存
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            gc.collect()

        print("🧹 CLIP_Surgery特征缓存已清理")

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        info = {
            'model_name': self.model_name,
            'scene_hash': self.scene_hash,
            'has_features': bool(self.feature_cache),
            'device': self.device
        }

        if self.feature_cache:
            point_features = self.feature_cache.get('point_features', {})
            info.update({
                'num_3d_points': len(point_features),
                'num_views': self.feature_cache.get('predictions_meta', {}).get('num_views', 0)
            })

        return info


class ClipSurgeryViserIntegration:
    """CLIP_Surgery与Viser的集成类"""

    def __init__(self, viser_server, clip_manager: ClipSurgeryFeatureManager, scene_center=None):
        """
        初始化Viser集成

        Args:
            viser_server: Viser服务器实例
            clip_manager: CLIP_Surgery特征管理器
            scene_center: 场景中心坐标，用于坐标对齐
        """
        self.viser_server = viser_server
        self.clip_manager = clip_manager
        self.highlight_objects = []  # 存储高亮对象的引用
        self.scene_center = scene_center if scene_center is not None else np.array([0, 0, 0])
        self.predictions = None  # 存储预测数据

        # 清除场景中可能存在的旧高亮对象
        self._clear_existing_highlights()

        # 设置GUI控件
        self.setup_gui()

        print("✅ CLIP_Surgery-Viser集成初始化完成")
        if scene_center is not None:
            print(f"   场景中心: [{self.scene_center[0]:.3f}, {self.scene_center[1]:.3f}, {self.scene_center[2]:.3f}]")

    def _clear_existing_highlights(self):
        """清除场景中可能存在的旧高亮对象"""
        if not self.viser_server:
            return

        try:
            # 尝试清除所有包含 "clip_highlight" 的节点
            if hasattr(self.viser_server.scene, '_nodes'):
                nodes_to_remove = []
                for path in self.viser_server.scene._nodes.keys():
                    if "clip_highlight" in path:
                        nodes_to_remove.append(path)

                removed_count = 0
                for path in nodes_to_remove:
                    try:
                        self.viser_server.scene._nodes[path].remove()
                        removed_count += 1
                    except Exception:
                        pass

                if removed_count > 0:
                    print(f"🧹 清除了 {removed_count} 个旧的高亮对象")

        except Exception as e:
            print(f"⚠️ 清除旧高亮对象时出现问题: {e}")

    def setup_gui(self):
        """设置Viser GUI控件"""
        if not self.viser_server:
            return

        # 文本输入框
        self.gui_text_query = self.viser_server.gui.add_text(
            "Query Text",
            initial_value="red chair"
        )

        # 相似度阈值滑块
        self.gui_similarity_threshold = self.viser_server.gui.add_slider(
            "Similarity Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=self.clip_manager.similarity_threshold
        )

        # 高亮按钮
        self.gui_highlight_button = self.viser_server.gui.add_button("Highlight Objects")

        # 清除高亮按钮
        self.gui_clear_button = self.viser_server.gui.add_button("Clear Highlights")

        # 状态显示
        self.gui_status = self.viser_server.gui.add_text(
            "Status",
            initial_value="Ready for highlighting (CLIP_Surgery)",
            disabled=True
        )

        # 绑定事件
        @self.gui_highlight_button.on_click
        def _(_):
            self.perform_highlight()

        @self.gui_clear_button.on_click
        def _(_):
            self.clear_highlights()

    def perform_highlight(self):
        """执行高亮操作"""
        try:
            # 检查特征是否已预计算
            if not self.clip_manager.feature_cache:
                self.gui_status.value = "Error: No features computed. Please run reconstruction first."
                return

            # 获取查询参数
            text_query = self.gui_text_query.value.strip()
            threshold = self.gui_similarity_threshold.value

            if not text_query:
                self.gui_status.value = "Error: Please enter a query text."
                return

            self.gui_status.value = f"Highlighting '{text_query}' (CLIP_Surgery)..."

            # 执行查询
            highlighted_points, highlighted_colors, stats = self.clip_manager.query_and_highlight(
                text_query, threshold
            )

            # 清除之前的高亮
            self.clear_highlights()

            # 显示高亮结果
            if len(highlighted_points) > 0:
                self.visualize_highlights(highlighted_points, highlighted_colors, text_query)

                # 更新状态
                self.gui_status.value = (
                    f"Found {len(highlighted_points)} points "
                    f"(max: {stats['max_similarity']:.3f}, "
                    f"mean: {stats['mean_similarity']:.3f})"
                )
            else:
                self.gui_status.value = f"No points found for '{text_query}' (threshold: {threshold:.3f})"

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.gui_status.value = error_msg
            print(f"❌ 高亮操作失败: {e}")
            import traceback
            traceback.print_exc()

    def visualize_highlights(self, points: np.ndarray, colors: np.ndarray, query_text: str):
        """在Viser中可视化高亮点"""
        try:
            # 应用场景中心化，与基础点云保持一致
            centered_points = points - self.scene_center

            # 添加高亮点云
            highlight_cloud = self.viser_server.scene.add_point_cloud(
                f"/clip_highlight_{query_text}",
                points=centered_points.astype(np.float32),
                colors=colors,
                point_size=self.clip_manager.point_size
            )
            self.highlight_objects.append(highlight_cloud)

            # 添加标签（在点云中心上方）
            if len(centered_points) > 0:
                center = np.mean(centered_points, axis=0)
                label_position = center + np.array([0, 0, 0.1])  # 稍微上移

                label = self.viser_server.scene.add_label(
                    f"/clip_highlight_{query_text}/label",
                    text=f"'{query_text}' ({len(points)} points)",
                    position=label_position
                )
                self.highlight_objects.append(label)

            print(f"✅ 高亮可视化完成: {len(points)} 个点")

        except Exception as e:
            print(f"❌ 高亮可视化失败: {e}")

    def clear_highlights(self):
        """清除所有高亮显示"""
        try:
            # 清除当前会话中跟踪的高亮对象
            removed_count = 0
            for obj in self.highlight_objects:
                try:
                    obj.remove()
                    removed_count += 1
                except Exception:
                    pass

            self.highlight_objects.clear()

            # 额外清除：尝试移除可能遗留的高亮对象
            try:
                if hasattr(self.viser_server.scene, '_nodes'):
                    nodes_to_remove = []
                    for path in self.viser_server.scene._nodes.keys():
                        if "clip_highlight" in path:
                            nodes_to_remove.append(path)

                    for path in nodes_to_remove:
                        try:
                            self.viser_server.scene._nodes[path].remove()
                            removed_count += 1
                        except Exception:
                            pass
            except Exception:
                pass

            self.gui_status.value = "Highlights cleared"
            if removed_count > 0:
                print(f"🧹 高亮显示已清除 ({removed_count} 个对象)")

        except Exception as e:
            print(f"⚠️ 清除高亮失败: {e}")

    def update_features(self, predictions: Dict[str, Any]):
        """更新特征（在重建完成后调用）"""
        try:
            # 存储预测数据
            self.predictions = predictions

            self.gui_status.value = "Computing CLIP_Surgery features..."

            # 预计算特征
            scene_hash = self.clip_manager.precompute_features(predictions)

            # 更新状态
            cache_info = self.clip_manager.get_cache_info()
            if cache_info.get('has_features', False):
                self.gui_status.value = (
                    f"CLIP_Surgery ready: {cache_info.get('num_3d_points', 0)} points, "
                    f"{cache_info.get('num_views', 0)} views"
                )
            else:
                self.gui_status.value = "CLIP_Surgery features ready"

            print(f"✅ CLIP_Surgery特征更新完成: {scene_hash}")

        except Exception as e:
            error_msg = f"Feature computation failed: {str(e)}"
            self.gui_status.value = error_msg
            print(f"❌ 特征更新失败: {e}")
            import traceback
            traceback.print_exc()


def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices is not None:
            device = "cuda:0"
        else:
            device = "cuda"

        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"🚀 使用GPU设备: {device} ({device_name})")
        return device
    else:
        print("⚠️ CUDA不可用，使用CPU")
        return "cpu"


def load_vggt_model(device):
    """加载VGGT模型"""
    print("🔄 正在加载VGGT模型...")
    start_time = time.time()

    model = VGGT.from_pretrained("/data22/ljc/proj/ckpt/VGGT-1B")
    model.eval()
    model = model.to(device)

    load_time = time.time() - start_time
    print(f"✅ VGGT模型加载完成 ({load_time:.2f}s)")

    return model


def load_or_reconstruct_scene(data_dir: str, model, device):
    """加载或重建场景"""
    data_path = Path(data_dir)

    # 检查是否有预存的重建结果
    predictions_file = data_path / "vggt_predictions.pkl"

    if predictions_file.exists():
        print(f"📂 加载预存的重建结果: {predictions_file}")
        with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)
        print("✅ 重建结果加载完成")
        return predictions

    # 如果没有预存结果，进行重建
    print(f"🔄 开始重建场景: {data_dir}")

    # 查找图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.HEIC']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f"**/*{ext}")))
        image_files.extend(list(data_path.glob(f"**/*{ext.upper()}")))

    if not image_files:
        raise ValueError(f"在 {data_dir} 中未找到图像文件")

    # 排序并限制数量（演示用）
    image_files = sorted(image_files)[:20]  # 最多20张图片
    image_paths = [str(f) for f in image_files]

    print(f"📷 找到 {len(image_paths)} 张图像")

    # 预处理图像
    print("🔄 预处理图像...")
    images = load_and_preprocess_images(image_paths)
    images = images.to(device)

    # 检查并修正图像张量维度
    print(f"📐 原始图像张量形状: {images.shape}")
    if len(images.shape) == 4:  # (S, C, H, W)
        # VGGT期望 (B, S, C, H, W) 格式，添加batch维度
        images = images.unsqueeze(0)  # (1, S, C, H, W)
        print(f"📐 修正后图像张量形状: {images.shape}")

    # VGGT重建
    print("🔄 VGGT重建中...")
    start_time = time.time()

    with torch.no_grad():
        predictions = model(images)

    reconstruction_time = time.time() - start_time
    print(f"✅ VGGT重建完成 ({reconstruction_time:.2f}s)")

    # 转换姿态编码为外参和内参矩阵
    print("🔄 转换姿态编码...")
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 处理模型输出：移除batch维度并转换为numpy
    print("🔄 处理模型输出...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # 移除batch维度

    print(f"📐 处理后的数据形状:")
    for key, value in predictions.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")

    # 保存重建结果
    try:
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"💾 重建结果已保存: {predictions_file}")
    except Exception as e:
        print(f"⚠️ 保存重建结果失败: {e}")

    return predictions


def visualize_basic_scene(viser_server, predictions):
    """可视化基础场景（点云和相机）"""
    print("🎨 可视化基础场景...")

    # 生成点云
    world_points = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )

    # 获取图像颜色
    images = predictions["images"]  # (S, 3, H, W)
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)

    S, H, W, _ = world_points.shape

    # 重塑为平面数据
    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)

    # 过滤有效点
    valid_mask = ~np.isnan(points_flat).any(axis=1)
    valid_mask &= ~np.isinf(points_flat).any(axis=1)

    valid_points = points_flat[valid_mask]
    valid_colors = colors_flat[valid_mask]

    if len(valid_points) > 0:
        # 计算场景中心并中心化
        scene_center = np.mean(valid_points, axis=0)
        centered_points = valid_points - scene_center

        # 添加点云
        point_cloud = viser_server.scene.add_point_cloud(
            "/point_cloud",
            points=centered_points.astype(np.float32),
            colors=valid_colors,
            point_size=0.001
        )

        print(f"✅ 已添加点云: {len(centered_points)} 个点")

        # 添加相机可视化
        visualize_cameras(viser_server, predictions, scene_center)

        return scene_center
    else:
        print("⚠️ 没有有效的点云数据")
        return np.array([0, 0, 0])


def visualize_cameras(viser_server, predictions, scene_center):
    """可视化相机位置和图像"""
    print("📷 添加相机可视化...")

    # 获取相机参数
    extrinsics = predictions["extrinsic"]  # (S, 3, 4)
    images = predictions["images"]  # (S, 3, H, W)

    # 转换相机外参为世界坐标
    cam_to_world_mat = closed_form_inverse_se3(extrinsics)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # 应用场景中心化
    cam_to_world[..., -1] -= scene_center

    frames = []
    frustums = []

    def attach_callback(frustum, frame):
        """为相机视锥体添加点击回调"""
        @frustum.on_click
        def _(_):
            for client in viser_server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    # 为每个相机添加可视化
    for img_id in range(len(images)):
        cam2world_3x4 = cam_to_world[img_id]
        T_world_camera = viser.transforms.SE3.from_matrix(cam2world_3x4)

        # 添加相机坐标轴
        frame_axis = viser_server.scene.add_frame(
            f"/cameras/frame_{img_id}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )
        frames.append(frame_axis)

        # 转换图片格式
        img = images[img_id]  # (3, H, W)
        img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
        h, w = img.shape[:2]

        # 计算FOV
        fy = 1.1 * h
        fov = 2 * np.arctan2(h / 2, fy)

        # 添加视锥体
        frustum_cam = viser_server.scene.add_camera_frustum(
            f"/cameras/frame_{img_id}/frustum",
            fov=fov,
            aspect=w / h,
            scale=0.05,
            image=img,
            line_width=1.0
        )
        frustums.append(frustum_cam)
        attach_callback(frustum_cam, frame_axis)

    print(f"✅ 已添加 {len(frames)} 个相机")

    # 添加GUI控制
    gui_show_cameras = viser_server.gui.add_checkbox("Show Cameras", initial_value=True)

    @gui_show_cameras.on_update
    def _(_):
        for frame in frames:
            frame.visible = gui_show_cameras.value
        for frustum in frustums:
            frustum.visible = gui_show_cameras.value

    return frames, frustums


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CLIP_Surgery语言引导3D高亮演示")
    parser.add_argument("--data_dir", type=str, required=True, help="重建数据目录")
    parser.add_argument("--port", type=int, default=6077, help="Viser服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser服务器主机")
    parser.add_argument("--model", type=str, default="CS-ViT-B/16", help="CLIP_Surgery模型")

    args = parser.parse_args()

    print("🚀 CLIP_Surgery语言引导3D高亮演示启动")
    print(f"📁 数据目录: {args.data_dir}")
    print(f"🌐 服务器: {args.host}:{args.port}")
    print(f"🤖 CLIP_Surgery模型: {args.model}")

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return

    # 获取设备
    device = get_device()

    # 加载VGGT模型
    vggt_model = load_vggt_model(device)

    # 加载或重建场景
    predictions = load_or_reconstruct_scene(args.data_dir, vggt_model, device)

    # 启动Viser服务器
    print(f"🚀 启动Viser服务器: {args.host}:{args.port}")
    viser_server = viser.ViserServer(host=args.host, port=args.port)
    viser_server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # 可视化基础场景
    scene_center = visualize_basic_scene(viser_server, predictions)

    # 初始化CLIP_Surgery特征管理器
    print("🔄 初始化CLIP_Surgery特征管理器...")
    clip_manager = ClipSurgeryFeatureManager(model_name=args.model)

    # 初始化CLIP_Surgery-Viser集成
    print("🔄 初始化CLIP_Surgery-Viser集成...")
    clip_integration = ClipSurgeryViserIntegration(viser_server, clip_manager, scene_center=scene_center)

    # 预计算CLIP_Surgery特征
    print("🔄 预计算CLIP_Surgery特征...")
    clip_integration.update_features(predictions)

    print(f"✅ CLIP_Surgery演示系统启动完成!")
    print(f"🌐 请访问: http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"🌐 或访问: http://localhost:{args.port}")
    print(f"💡 使用说明:")
    print(f"   1. 在'Query Text'中输入查询文本（如'red chair', 'plant', 'table'）")
    print(f"   2. 调整'Similarity Threshold'滑块设置相似度阈值")
    print(f"   3. 点击'Highlight Objects'按钮进行高亮")
    print(f"   4. 点击'Clear Highlights'按钮清除高亮")
    print(f"🔧 按Ctrl+C退出")

    try:
        # 保持服务器运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 CLIP_Surgery演示系统关闭")


if __name__ == "__main__":
    main()
