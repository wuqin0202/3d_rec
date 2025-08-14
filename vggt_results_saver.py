#!/usr/bin/env python3
"""VGGT Results Saver and Loader

This module provides functions to save and load VGGT inference results
including camera intrinsics, extrinsics, and depth maps to/from HDF5 files.

Author: Assistant
Date: 2025-08-13
"""

import os
import logging
import h5py
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 extrinsic matrices to 4x4 homogeneous matrices.

    Args:
        extrinsics: Camera extrinsic parameters with shape [B, S, 3, 4]

    Returns:
        Homogeneous extrinsic matrices with shape [B, S, 4, 4]
    """
    batch, seq = extrinsics.shape[:2]
    last_row = torch.tensor([0, 0, 0, 1], device=extrinsics.device).reshape(1, 1, 1, 4)
    last_row = last_row.expand(batch, seq, -1, -1)
    return torch.cat([extrinsics, last_row], dim=2)


class VGGTResultsSaver:
    """Class for saving and loading VGGT inference results."""

    def __init__(self, model_path: str, device: str = 'auto', dtype: str = 'auto'):
        """Initialize the VGGT model and saver.

        Args:
            model_path: Path to the pretrained VGGT model
            device: Device to use ('auto', 'cuda', 'cpu')
            dtype: Data type to use ('auto', 'bfloat16', 'float16', 'float32')
        """
        self.device = self._setup_device(device)
        self.dtype = self._setup_dtype(dtype)
        self.model = self._load_model(model_path)

        logger.info(f'Initialized VGGT with device: {self.device}, dtype: {self.dtype}')

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _setup_dtype(self, dtype: str) -> torch.dtype:
        """Setup data type."""
        if dtype == 'auto':
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
            if self.device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
                return torch.bfloat16
            else:
                return torch.float16
        elif dtype == 'bfloat16':
            return torch.bfloat16
        elif dtype == 'float16':
            return torch.float16
        elif dtype == 'float32':
            return torch.float32
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')

    def _load_model(self, model_path: str) -> VGGT:
        """Load the VGGT model."""
        logger.info(f'Loading VGGT model from: {model_path}')
        model = VGGT.from_pretrained(model_path).to(self.device)
        model.eval()
        return model

    def run_inference(self, image_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Run VGGT inference on a list of images.

        Args:
            image_paths: List of paths to input images

        Returns:
            Dictionary containing VGGT predictions
        """
        logger.info(f'Running inference on {len(image_paths)} images')

        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)
        logger.info(f'Input images shape: {images.shape}')

        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)

        logger.info(f'Inference completed. Available keys: {list(predictions.keys())}')
        return predictions

    def extract_camera_parameters(self, predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract camera intrinsics and extrinsics from predictions.

        Args:
            predictions: VGGT predictions dictionary

        Returns:
            Tuple of (extrinsics_4x4, intrinsics)
        """
        # Get image size from depth map
        image_size_hw = predictions['depth_conf'].shape[2:4]  # (H, W)

        # Extract camera parameters
        extrinsics_3x4, intrinsics = pose_encoding_to_extri_intri(
            predictions['pose_enc'],
            image_size_hw
        )

        # Convert to 4x4 homogeneous matrices
        extrinsics_4x4 = to_homogeneous(extrinsics_3x4)

        logger.info(f'Camera parameters extracted:')
        logger.info(f'  Extrinsics 4x4 shape: {extrinsics_4x4.shape}')
        logger.info(f'  Intrinsics shape: {intrinsics.shape}')

        return extrinsics_4x4, intrinsics

    def save_results(
        self,
        predictions: Dict[str, torch.Tensor],
        image_paths: List[str],
        output_path: str,
        save_additional_outputs: bool = True
    ) -> None:
        """Save VGGT inference results to HDF5 file.

        Args:
            predictions: VGGT predictions dictionary
            image_paths: List of input image paths
            output_path: Path to save the HDF5 file
            save_additional_outputs: Whether to save additional outputs like world points
        """
        logger.info(f'Saving results to: {output_path}')

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Extract camera parameters
        extrinsics_4x4, intrinsics = self.extract_camera_parameters(predictions)

        # Convert tensors to numpy arrays for saving
        def tensor_to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        with h5py.File(output_path, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['num_images'] = len(image_paths)
            metadata_group.attrs['image_height'] = predictions['depth'].shape[2]
            metadata_group.attrs['image_width'] = predictions['depth'].shape[3]
            metadata_group.attrs['batch_size'] = predictions['depth'].shape[0]
            metadata_group.attrs['sequence_length'] = predictions['depth'].shape[1]

            # Save image paths and indices
            image_paths_encoded = [path.encode('utf-8') for path in image_paths]
            metadata_group.create_dataset('image_paths', data=image_paths_encoded)
            # Save image indices (0-based indexing)
            image_indices = np.arange(len(image_paths))
            metadata_group.create_dataset('image_indices', data=image_indices)

            # Save camera parameters
            camera_group = f.create_group('camera_parameters')
            camera_group.create_dataset('extrinsics_4x4', data=tensor_to_numpy(extrinsics_4x4))
            camera_group.create_dataset('intrinsics', data=tensor_to_numpy(intrinsics))
            camera_group.create_dataset('pose_encoding', data=tensor_to_numpy(predictions['pose_enc']))
            # Save image indices for camera parameters (same as metadata but for convenience)
            camera_group.create_dataset('image_indices', data=image_indices)

            # Save depth maps
            depth_group = f.create_group('depth')
            depth_group.create_dataset('depth_map', data=tensor_to_numpy(predictions['depth']))
            depth_group.create_dataset('depth_confidence', data=tensor_to_numpy(predictions['depth_conf']))
            # Save image indices for depth maps (same as metadata but for convenience)
            depth_group.create_dataset('image_indices', data=image_indices)

            # Save additional outputs if requested
            if save_additional_outputs:
                additional_group = f.create_group('additional_outputs')

                if 'world_points' in predictions:
                    additional_group.create_dataset('world_points', data=tensor_to_numpy(predictions['world_points']))

                if 'world_points_conf' in predictions:
                    additional_group.create_dataset('world_points_confidence', data=tensor_to_numpy(predictions['world_points_conf']))

                if 'pose_enc_list' in predictions:
                    pose_enc_list_group = additional_group.create_group('pose_enc_list')
                    for i, pose_enc in enumerate(predictions['pose_enc_list']):
                        pose_enc_list_group.create_dataset(f'level_{i}', data=tensor_to_numpy(pose_enc))

        logger.info(f'Results saved successfully to: {output_path}')

    def process_and_save(
        self,
        image_paths: List[str],
        output_path: str,
        save_additional_outputs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Process images and save results in one step.

        Args:
            image_paths: List of paths to input images
            output_path: Path to save the HDF5 file
            save_additional_outputs: Whether to save additional outputs

        Returns:
            VGGT predictions dictionary
        """
        # Run inference
        predictions = self.run_inference(image_paths)

        # Save results
        self.save_results(predictions, image_paths, output_path, save_additional_outputs)

        return predictions


def load_vggt_results(file_path: str) -> Dict[str, Union[np.ndarray, List[str]]]:
    """Load VGGT inference results from HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        Dictionary containing loaded results with the following structure:
        {
            'metadata': {
                'num_images': int,
                'image_height': int,
                'image_width': int,
                'batch_size': int,
                'sequence_length': int,
                'image_paths': List[str],
                'image_indices': np.ndarray  # Array of image indices (0-based)
            },
            'camera_parameters': {
                'extrinsics_4x4': np.ndarray,  # Shape: [B, S, 4, 4]
                'intrinsics': np.ndarray,      # Shape: [B, S, 3, 3]
                'pose_encoding': np.ndarray,   # Shape: [B, S, 9]
                'image_indices': np.ndarray    # Array of image indices (0-based)
            },
            'depth': {
                'depth_map': np.ndarray,       # Shape: [B, S, H, W]
                'depth_confidence': np.ndarray, # Shape: [B, S, H, W]
                'image_indices': np.ndarray    # Array of image indices (0-based)
            },
            'additional_outputs': {  # Optional, if saved
                'world_points': np.ndarray,           # Shape: [B, S, H, W, 3]
                'world_points_confidence': np.ndarray, # Shape: [B, S, H, W]
                'pose_enc_list': List[np.ndarray]     # List of pose encodings at different levels
            }
        }
    """
    logger.info(f'Loading VGGT results from: {file_path}')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    results = {}

    with h5py.File(file_path, 'r') as f:
        # Load metadata
        metadata = {}
        metadata_group = f['metadata']
        for key in metadata_group.attrs.keys():
            metadata[key] = metadata_group.attrs[key]

        # Decode image paths
        image_paths_encoded = metadata_group['image_paths'][:]
        metadata['image_paths'] = [path.decode('utf-8') for path in image_paths_encoded]
        # Load image indices if available
        if 'image_indices' in metadata_group:
            metadata['image_indices'] = metadata_group['image_indices'][:]
        results['metadata'] = metadata

        # Load camera parameters
        camera_params = {}
        camera_group = f['camera_parameters']
        for key in camera_group.keys():
            camera_params[key] = camera_group[key][:]
        results['camera_parameters'] = camera_params

        # Load depth maps
        depth_data = {}
        depth_group = f['depth']
        for key in depth_group.keys():
            depth_data[key] = depth_group[key][:]
        results['depth'] = depth_data

        # Load additional outputs if they exist
        if 'additional_outputs' in f:
            additional_outputs = {}
            additional_group = f['additional_outputs']

            for key in additional_group.keys():
                if key == 'pose_enc_list':
                    # Special handling for pose_enc_list
                    pose_enc_list = []
                    pose_enc_list_group = additional_group[key]
                    level_keys = sorted(pose_enc_list_group.keys(), key=lambda x: int(x.split('_')[1]))
                    for level_key in level_keys:
                        pose_enc_list.append(pose_enc_list_group[level_key][:])
                    additional_outputs[key] = pose_enc_list
                else:
                    additional_outputs[key] = additional_group[key][:]

            results['additional_outputs'] = additional_outputs

    logger.info(f'Results loaded successfully from: {file_path}')
    logger.info(f'Loaded {results["metadata"]["num_images"]} images')

    return results


def get_camera_matrices_for_image(
    results: Dict[str, Union[np.ndarray, List[str]]],
    image_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get camera matrices for a specific image.

    Args:
        results: Loaded VGGT results dictionary
        image_index: Index of the image (0-based)

    Returns:
        Tuple of (extrinsics_4x4, intrinsics) for the specified image
    """
    if image_index >= results['metadata']['num_images']:
        raise IndexError(f'Image index {image_index} out of range. Total images: {results["metadata"]["num_images"]}')

    # Assuming batch_size=1 and the image_index corresponds to sequence dimension
    batch_idx = 0
    seq_idx = image_index

    extrinsics_4x4 = results['camera_parameters']['extrinsics_4x4'][batch_idx, seq_idx]
    intrinsics = results['camera_parameters']['intrinsics'][batch_idx, seq_idx]

    return extrinsics_4x4, intrinsics


def get_depth_map_for_image(
    results: Dict[str, Union[np.ndarray, List[str]]],
    image_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get depth map and confidence for a specific image.

    Args:
        results: Loaded VGGT results dictionary
        image_index: Index of the image (0-based)

    Returns:
        Tuple of (depth_map, depth_confidence) for the specified image
    """
    if image_index >= results['metadata']['num_images']:
        raise IndexError(f'Image index {image_index} out of range. Total images: {results["metadata"]["num_images"]}')

    # Assuming batch_size=1 and the image_index corresponds to sequence dimension
    batch_idx = 0
    seq_idx = image_index

    depth_map = results['depth']['depth_map'][batch_idx, seq_idx]
    depth_confidence = results['depth']['depth_confidence'][batch_idx, seq_idx]

    return depth_map, depth_confidence


def get_image_info_by_index(
    results: Dict[str, Union[np.ndarray, List[str]]],
    image_index: int
) -> Dict[str, Union[str, int, np.ndarray]]:
    """Get all information for a specific image by index.

    Args:
        results: Loaded VGGT results dictionary
        image_index: Index of the image (0-based)

    Returns:
        Dictionary containing all information for the specified image:
        {
            'image_path': str,
            'image_index': int,
            'extrinsics_4x4': np.ndarray,
            'intrinsics': np.ndarray,
            'depth_map': np.ndarray,
            'depth_confidence': np.ndarray
        }
    """
    if image_index >= results['metadata']['num_images']:
        raise IndexError(f'Image index {image_index} out of range. Total images: {results["metadata"]["num_images"]}')

    # Get camera matrices and depth maps
    extrinsics_4x4, intrinsics = get_camera_matrices_for_image(results, image_index)
    depth_map, depth_confidence = get_depth_map_for_image(results, image_index)

    return {
        'image_path': results['metadata']['image_paths'][image_index],
        'image_index': image_index,
        'extrinsics_4x4': extrinsics_4x4,
        'intrinsics': intrinsics,
        'depth_map': depth_map,
        'depth_confidence': depth_confidence
    }


def get_image_info_by_path(
    results: Dict[str, Union[np.ndarray, List[str]]],
    image_path: str
) -> Dict[str, Union[str, int, np.ndarray]]:
    """Get all information for a specific image by path.

    Args:
        results: Loaded VGGT results dictionary
        image_path: Path to the image

    Returns:
        Dictionary containing all information for the specified image
    """
    try:
        image_index = results['metadata']['image_paths'].index(image_path)
    except ValueError:
        raise ValueError(f'Image path not found: {image_path}')

    return get_image_info_by_index(results, image_index)


def list_all_images(results: Dict[str, Union[np.ndarray, List[str]]]) -> List[Dict[str, Union[str, int]]]:
    """List all images with their indices and paths.

    Args:
        results: Loaded VGGT results dictionary

    Returns:
        List of dictionaries with image information:
        [{'index': int, 'path': str}, ...]
    """
    return [
        {'index': i, 'path': path}
        for i, path in enumerate(results['metadata']['image_paths'])
    ]


# Example usage

# Example script demonstrating how to use the VGGT Results Saver.
# This script shows how to:
# 1. Process images with VGGT
# 2. Save results to HDF5 file
# 3. Load and access the saved results
import sys
import argparse

def main():
    """命令行参数方式运行VGGT结果保存与加载示例。"""
    parser = argparse.ArgumentParser(description='VGGT Results Saver Example')
    parser.add_argument('--model_path', type=str, required=True, help='VGGT模型权重路径')
    parser.add_argument('--images_dir', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--output_file', type=str, required=True, help='输出HDF5文件路径')
    parser.add_argument('--device', type=str, default='auto', help='推理设备，auto/cuda/cpu')
    parser.add_argument('--dtype', type=str, default='auto', help='推理精度，auto/float16/float32/bfloat16')
    parser.add_argument('--suffix', type=str, default='.jpg', help='图片后缀名，默认.jpg')
    parser.add_argument('--save_additional_outputs', action='store_true', help='保存额外输出')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='设置CUDA_VISIBLE_DEVICES')
    parser.add_argument('--home', type=str, default=None, help='设置HOME环境变量')
    args = parser.parse_args()

    # 环境变量设置
    if args.cuda_visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    if args.home:
        os.environ['HOME'] = args.home

    logger.info('Starting VGGT results processing example')

    # Step 1: Initialize the saver
    logger.info('Initializing VGGT saver...')
    saver = VGGTResultsSaver(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype
    )

    # Step 2: Get image paths
    logger.info(f'Collecting images from: {args.images_dir}')
    image_paths = [
        os.path.join(args.images_dir, f)
        for f in sorted(os.listdir(args.images_dir))
        if f.endswith(args.suffix)
    ]

    if not image_paths:
        logger.error(f'No images found in {args.images_dir} with suffix {args.suffix}')
        exit(1)

    logger.info(f'Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths[:5]]}...')

    # Step 3: Process images and save results
    logger.info('Processing images with VGGT and saving results...')
    predictions = saver.process_and_save(
        image_paths=image_paths,
        output_path=args.output_file,
        save_additional_outputs=args.save_additional_outputs
    )

    # Step 4: Load and verify the saved results
    logger.info('Loading saved results...')
    loaded_results = load_vggt_results(args.output_file)

    # Step 5: Display summary information
    logger.info('=== Results Summary ===')
    metadata = loaded_results['metadata']
    print(f'Number of images: {metadata["num_images"]}')
    print(f'Image resolution: {metadata["image_height"]}x{metadata["image_width"]}')
    print(f'Batch size: {metadata["batch_size"]}')
    print(f'Sequence length: {metadata["sequence_length"]}')
    print(f'Image paths: {metadata["image_paths"][:3]}...')
    if 'image_indices' in metadata:
        print(f'Image indices available: {metadata["image_indices"][:5]}...')

    # Step 5.5: List all images with indices
    logger.info('=== All Images List ===')
    all_images = list_all_images(loaded_results)
    for img_info in all_images[:5]:  # Show first 5
        print(f'Index {img_info["index"]}: {os.path.basename(img_info["path"])}')
    if len(all_images) > 5:
        print(f'... and {len(all_images) - 5} more images')

    # Step 6: Access camera parameters for individual images
    logger.info('=== Camera Parameters for First Image ===')
    extrinsics_4x4, intrinsics = get_camera_matrices_for_image(loaded_results, 0)

    print(f'Extrinsics 4x4 shape: {extrinsics_4x4.shape}')
    print(f'Intrinsics shape: {intrinsics.shape}')

    print('\nIntrinsics matrix (first image):')
    print(intrinsics)

    print('\nExtrinsics 4x4 matrix (first image):')
    print(extrinsics_4x4)

    # Step 7: Access depth maps for individual images
    logger.info('=== Depth Information for First Image ===')
    depth_map, depth_confidence = get_depth_map_for_image(loaded_results, 0)

    print(f'Depth map shape: {depth_map.shape}')
    print(f'Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]')
    print(f'Depth confidence shape: {depth_confidence.shape}')
    print(f'Depth confidence range: [{depth_confidence.min():.3f}, {depth_confidence.max():.3f}]')

    # Step 7.5: Get complete image information by index
    logger.info('=== Complete Image Information (Index 0) ===')
    image_info = get_image_info_by_index(loaded_results, 0)
    print(f'Image path: {os.path.basename(image_info["image_path"])}')
    print(f'Image index: {image_info["image_index"]}')
    print(f'Extrinsics shape: {image_info["extrinsics_4x4"].shape}')
    print(f'Intrinsics shape: {image_info["intrinsics"].shape}')
    print(f'Depth map shape: {image_info["depth_map"].shape}')
    print(f'Depth confidence shape: {image_info["depth_confidence"].shape}')

    # Step 8: Show available data groups
    logger.info('=== Available Data Groups ===')
    for group_name, group_data in loaded_results.items():
        if isinstance(group_data, dict):
            print(f'{group_name}:')
            for key in group_data.keys():
                if hasattr(group_data[key], 'shape'):
                    print(f'  {key}: shape {group_data[key].shape}')
                else:
                    print(f'  {key}: {type(group_data[key])}')

    logger.info('Example completed successfully!')

if __name__ == '__main__':
    main()
