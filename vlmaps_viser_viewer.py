#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已保存的 VLMaps 3D 体素特征图进行 viser 可视化与文本高亮。

功能：
- 加载 output/vlmaps_features_clip_patch.h5df（由 mapping_utils.save_3d_map 保存）
- 可视化点云（使用保存的 grid_rgb 作为颜色，若无则使用默认颜色）
- 支持文本查询，基于 CLIP_Surgery 文本-视觉相似度高亮对应的点

用法：
- 直接运行（假设默认输出路径与默认参数）：
  python vlmaps_viser_viewer.py

- 指定文件与参数：
  python vlmaps_viser_viewer.py \
    --map output/vlmaps_features_clip_patch.h5df \
    --cell_size 0.05 \
    --camera_height 3.0 \
    --host 0.0.0.0 --port 6080

注意：
- H5DF 未存储 cell_size/camera_height，需通过参数提供；若未知，默认采用 main.py 中的默认值（cell_size=0.05, camera_height=3.0）。
- 代码风格：单引号与 logging，Google 风格 Docstring。
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np

import viser
import viser.transforms

import torch

import clip  # 本仓库的 CLIP_Surgery 模块
from vlmaps.utils.mapping_utils import (
    load_3d_map,
    grid_id2base_pos_3d_batch,
)


# -----------------------------------------------------------------------------
# logging 配置
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CLIP 文本编码与相似度
# -----------------------------------------------------------------------------

class ClipTextEncoder:
    """CLIP_Surgery 文本编码器与相似度计算。

    Attributes:
        device: 'cuda' 或 'cpu'
        model: 已加载的 CLIP_Surgery 模型
    """

    def __init__(self, device: Optional[str] = None, model_name: str = 'CS-ViT-B/16') -> None:
        if device in (None, 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 延迟加载模型，避免无查询也占资源
        self._model = None
        self._preprocess = None
        self._model_name = model_name

    @property
    def model(self):
        if self._model is None:
            logger.info('Loading CLIP_Surgery model: %s', self._model_name)
            self._model, self._preprocess = clip.load(self._model_name, device=self.device)
            self._model = self._model.eval()
        return self._model

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """编码单条文本为 L2 归一化特征。

        Args:
            text: 查询文本

        Returns:
            torch.Tensor: (C,) 归一化特征
        """
        # 优先使用仓库提供的 prompt ensemble 方法
        text_feat = clip.encode_text_with_prompt_ensemble(self.model, [text], self.device)[0]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat


# -----------------------------------------------------------------------------
# 可视化应用
# -----------------------------------------------------------------------------

class VLMAPSViserApp:
    """基于保存的 3D 体素特征图进行 viser 可视化与文本高亮。"""

    def __init__(
        self,
        map_path: str = 'output/vlmaps_features_clip_patch.h5df',
        cell_size: float = 0.05,
        camera_height: float = 3.0,
        host: str = '0.0.0.0',
        port: int = 6080,
        max_points: int = 300000,
        device: str = 'auto',
    ) -> None:
        """初始化应用。

        Args:
            map_path: H5DF 文件路径（save_3d_map 保存的结果）
            cell_size: 体素边长（米），用于将 grid_pos 恢复为 base 坐标
            camera_height: 相机高度（米），用于 3D 网格坐标到 base 坐标的转换
            host: viser 服务器 host
            port: viser 服务器端口
            max_points: 最大显示的点数（超过将随机下采样）
            device: 文本编码设备
        """
        self.map_path = map_path
        self.cell_size = float(cell_size)
        self.camera_height = float(camera_height)
        self.host = host
        self.port = int(port)
        self.max_points = int(max_points)
        self.device = device

        # 数据容器
        self.grid_feat = None  # (N, C)
        self.grid_pos = None   # (N, 3) -> (row, col, height)
        self.grid_rgb = None   # (N, 3) uint8
        self.occupied_shape = None  # (gs, gs, vh)

        # 点云渲染缓存
        self.points_xyz = None  # (N, 3) float32（中心化）
        self.points_rgb = None  # (N, 3) uint8
        self.scene_center = None  # (3,)

        # 高亮对象句柄
        self._highlight_nodes = []
        self._base_points_node = None  # 基础点云句柄

        # 模型与服务器
        self.text_encoder = ClipTextEncoder(device=device)
        self.server = None

    # -------------------------- 数据加载与预处理 --------------------------
    def load_map(self) -> None:
        """加载 H5DF 地图文件，并构建点云。"""
        logger.info('Loading 3D voxel map: %s', self.map_path)
        mapped_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb = load_3d_map(self.map_path)
        self.grid_feat = grid_feat.astype(np.float32)
        self.grid_pos = grid_pos.astype(np.int32)
        self.grid_rgb = grid_rgb.astype(np.uint8) if grid_rgb is not None else None
        self.occupied_shape = tuple(occupied_ids.shape)
        gs = self.occupied_shape[0]
        logger.info('Map loaded: N=%d, feat_dim=%d, gs=%d, vh=%d', grid_feat.shape[0], grid_feat.shape[1], gs, self.occupied_shape[2])

        # 将 (row, col, height) -> base 坐标 (x_base, y_base, z_base)
        # 注意：grid_id2base_pos_3d_batch 返回 [base_x, base_y, base_z] 列表
        base_xyz_list = grid_id2base_pos_3d_batch(self.grid_pos, self.cell_size, gs)
        base_xyz = np.stack(base_xyz_list, axis=1).astype(np.float32)

        # 计算场景中心并中心化
        center = np.mean(base_xyz, axis=0)
        self.scene_center = center.astype(np.float32)
        centered_xyz = (base_xyz - self.scene_center[None, :]).astype(np.float32)

        # 颜色
        if self.grid_rgb is None:
            colors = np.full((centered_xyz.shape[0], 3), 200, dtype=np.uint8)
        else:
            colors = self.grid_rgb

        # 下采样（防止点数过大导致渲染缓慢）
        n = centered_xyz.shape[0]
        if self.max_points > 0 and n > self.max_points:
            logger.info('Downsampling points: %d -> %d', n, self.max_points)
            idx = np.random.choice(n, size=self.max_points, replace=False)
            centered_xyz = centered_xyz[idx]
            colors = colors[idx]
            if self.grid_feat is not None:
                self.grid_feat = self.grid_feat[idx]

        self.points_xyz = centered_xyz
        self.points_rgb = colors

    # --------------------------- viser 场景构建 ---------------------------
    def start_server(self) -> None:
        """启动 viser 服务器并构建 GUI。"""
        logger.info('Starting viser server at %s:%d', self.host, self.port)
        self.server = viser.ViserServer(host=self.host, port=self.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout='collapsible')

        # 添加基础点云
        if self.points_xyz is not None and self.points_xyz.shape[0] > 0:
            # 默认基础点大小
            base_pt_size = 0.05
            self._base_points_node = self.server.scene.add_point_cloud(
                '/vlmaps/points',
                points=self.points_xyz.astype(np.float32),
                colors=self.points_rgb,
                point_size=base_pt_size,
            )
            logger.info('Point cloud added: %d points', self.points_xyz.shape[0])
        else:
            logger.warning('No points to render.')

        # GUI 控件
        self._build_gui()

    def _build_gui(self) -> None:
        assert self.server is not None

        # 文本输入 & 阈值/TopK
        gui_text = self.server.gui.add_text('Query Text', initial_value='chair')
        gui_thresh = self.server.gui.add_slider('Similarity Threshold', min=0.0, max=1.0, step=0.01, initial_value=0.25)
        gui_topk = self.server.gui.add_slider('Top-K (0=off)', min=0, max=200000, step=100, initial_value=0)
        gui_pt_size = self.server.gui.add_slider('Highlight Point Size', min=0.005, max=0.05, step=0.001, initial_value=0.005)
        gui_base_pt_size = self.server.gui.add_slider('Base Point Size', min=0.005, max=0.05, step=0.001, initial_value=(self._base_points_node.point_size if self._base_points_node is not None else 0.005))
        btn_run = self.server.gui.add_button('Run Query')
        btn_clear = self.server.gui.add_button('Clear Highlights')
        txt_status = self.server.gui.add_text('Status', initial_value='Ready', disabled=True)

        @btn_run.on_click
        def _(_):
            try:
                query = gui_text.value.strip()
                if not query:
                    txt_status.value = 'Empty query.'
                    return
                threshold = float(gui_thresh.value)
                topk = int(gui_topk.value)
                point_size = float(gui_pt_size.value)
                n_hl, max_sim, mean_sim = self.highlight_query(query, threshold=threshold, topk=topk, point_size=point_size)
                txt_status.value = f"Highlighted: {n_hl} (max={max_sim:.3f}, mean={mean_sim:.3f})"
            except Exception as e:
                logger.exception('Query failed')
                txt_status.value = f'Error: {e}'

        @btn_clear.on_click
        def _(_):
            self.clear_highlights()
            txt_status.value = 'Highlights cleared'

        @gui_base_pt_size.on_update
        def _(_):
            # 实时调整基础点云大小
            try:
                if self._base_points_node is not None:
                    self._base_points_node.point_size = float(gui_base_pt_size.value)
            except Exception:
                logger.exception('Failed to update base point size')

        @gui_pt_size.on_update
        def _(_):
            # 同步更新已存在的高亮点云大小
            try:
                for n in self._highlight_nodes:
                    try:
                        n.point_size = float(gui_pt_size.value)
                    except Exception:
                        pass
            except Exception:
                logger.exception('Failed to update highlight point size')

    # --------------------------- 高亮逻辑 ---------------------------
    def highlight_query(self, text: str, threshold: float = 0.25, topk: int = 0, point_size: float = 0.003) -> Tuple[int, float, float]:
        """执行文本查询并高亮点云。

        Args:
            text: 查询文本
            threshold: 相似度阈值（与 topk 同时设置时，先按 topk 截断再应用阈值）
            topk: 选择相似度最高的前 K 个点（0 表示不启用）
            point_size: 高亮点大小

        Returns:
            (高亮点数量, 最大相似度, 平均相似度)
        """
        assert self.grid_feat is not None and self.points_xyz is not None

        # 文本编码
        text_feat = self.text_encoder.encode_text(text)
        text_feat_np = text_feat.detach().cpu().to(torch.float32).numpy().reshape(-1)

        # 归一化点特征
        feats = self.grid_feat
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)

        # 相似度
        sims = feats @ text_feat_np  # (N,)

        # Top-K（可选）
        idx = np.arange(sims.shape[0])
        if topk > 0 and topk < sims.shape[0]:
            topk_idx = np.argpartition(-sims, topk)[:topk]
            idx = topk_idx
            sims = sims[topk_idx]

        # 阈值过滤
        keep = sims >= float(threshold)
        if not np.any(keep):
            self.clear_highlights()
            return 0, float(np.max(sims)) if sims.size > 0 else 0.0, 0.0

        sel_idx = idx[keep]
        sel_pts = self.points_xyz[sel_idx]
        sel_scores = sims[keep]

        # 颜色映射：红->黄 根据相似度
        colors = self._scores_to_colors(sel_scores)

        # 渲染
        self.clear_highlights()
        node = self.server.scene.add_point_cloud(
            f'/vlmaps/highlight/{text}',
            points=sel_pts.astype(np.float32),
            colors=colors,
            point_size=point_size,
        )
        self._highlight_nodes.append(node)

        return sel_pts.shape[0], float(np.max(sel_scores)), float(np.mean(sel_scores))

    def clear_highlights(self) -> None:
        for n in self._highlight_nodes:
            try:
                n.remove()
            except Exception:
                pass
        self._highlight_nodes = []

    @staticmethod
    def _scores_to_colors(scores: np.ndarray) -> np.ndarray:
        """ 分数越高越黄 """
        if scores.size == 0:
            return np.empty((0, 3), dtype=np.uint8)
        s = scores.astype(np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-6)
        # 红(255,0,0) -> 黄(255,255,0)
        r = np.full_like(s, 255.0)
        g = 255.0 * s
        b = np.zeros_like(s)
        colors = np.stack([r, g, b], axis=1)
        return colors.clip(0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description='VLMaps viser viewer with text highlight')
    p.add_argument('--map', type=str, default='output/vlmaps_features_clip_patch.h5df', help='Path to H5DF map file')
    p.add_argument('--cell_size', type=float, default=0.05, help='Voxel cell size in meters')
    p.add_argument('--camera_height', type=float, default=3.0, help='Camera height in meters (for completeness)')
    p.add_argument('--host', type=str, default='0.0.0.0', help='Viser host')
    p.add_argument('--port', type=int, default=6080, help='Viser port')
    p.add_argument('--max_points', type=int, default=300000, help='Max number of points to render (downsample if exceeded)')
    p.add_argument('--device', type=str, default='auto', help='Device for text encoder')
    return p.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    app = VLMAPSViserApp(
        map_path=args.map,
        cell_size=args.cell_size,
        camera_height=args.camera_height,
        host=args.host,
        port=args.port,
        max_points=args.max_points,
        device=args.device,
    )
    app.load_map()
    app.start_server()

    logger.info('Viewer ready: http://%s:%d', args.host, args.port)
    if args.host == '0.0.0.0':
        logger.info('Or visit: http://localhost:%d', args.port)

    try:
        import time as _time
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Shutting down...')


if __name__ == '__main__':
    main()
