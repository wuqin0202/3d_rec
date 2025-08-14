#!/usr/bin/env python3
"""Example script demonstrating how to use the VGGT Results Saver.

This script shows how to:
1. Process images with VGGT
2. Save results to HDF5 file
3. Load and access the saved results
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.getcwd())

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['HOME'] = '/data25/wuqin'

from vggt_results_saver import VGGTResultsSaver, load_vggt_results, get_camera_matrices_for_image, get_depth_map_for_image, get_image_info_by_index, list_all_images

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating VGGT results saving and loading."""

    # Configuration
    model_path = '/data25/wuqin/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9'
    images_dir = 'examples/hall_4f_25'
    output_file = 'output/vggt_hall_4f_results.h5'

    logger.info('Starting VGGT results processing example')

    # Step 1: Initialize the saver
    logger.info('Initializing VGGT saver...')
    saver = VGGTResultsSaver(
        model_path=model_path,
        device='auto',  # Will auto-detect CUDA availability
        dtype='auto'    # Will auto-select optimal dtype
    )

    # Step 2: Get image paths
    logger.info(f'Collecting images from: {images_dir}')
    image_paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.endswith('.jpg')
    ]

    if not image_paths:
        logger.error(f'No JPG images found in {images_dir}')
        return

    logger.info(f'Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths[:5]]}...')

    # Step 3: Process images and save results
    logger.info('Processing images with VGGT and saving results...')
    predictions = saver.process_and_save(
        image_paths=image_paths,
        output_path=output_file,
        save_additional_outputs=True
    )

    # Step 4: Load and verify the saved results
    logger.info('Loading saved results...')
    loaded_results = load_vggt_results(output_file)

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

    print('\\nIntrinsics matrix (first image):')
    print(intrinsics)

    print('\\nExtrinsics 4x4 matrix (first image):')
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
