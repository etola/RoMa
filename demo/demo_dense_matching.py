#!/usr/bin/env python3
"""
Dense Matching with RoMa and COLMAP Integration

This script performs dense matching between image pairs using RoMa matcher,
with automatic pair selection based on COLMAP calibration data.
Generates point clouds from matching results and merges them.

Usage:
    # Basic usage - automatically uses scene_dir/images, scene_dir/sparse, scene_dir/output
    python demo_dense_matching.py -s /path/to/scene
    
    # Or using long flags:
    python demo_dense_matching.py --scene_dir /path/to/scene
    
    # Custom output directory (relative to scene_dir):
    python demo_dense_matching.py -s /path/to/scene -o my_results
    
    # Override individual paths if needed:
    python demo_dense_matching.py -s /path/to/scene \
                                  -c /path/to/custom/colmap \
                                  -i /path/to/custom/images

JSON Configuration Support:
    # Create an example configuration file:
    python demo_dense_matching.py --create_example_config config_example.json
    
    # Run with JSON configuration:
    python demo_dense_matching.py -j my_config.json
    
    # Override JSON values with command line arguments:
    python demo_dense_matching.py -j my_config.json -s /different/scene -v
    
    # The final configuration is automatically saved as run_config.json in the output directory
    # You can reuse this saved configuration to reproduce results:
    python demo_dense_matching.py -j /path/to/output/run_config.json
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ExifTags import TAGS
import open3d as o3d

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from colmap_utils import COLMAPDataset, triangulate_points
from romatch import roma_outdoor, tiny_roma_v1_outdoor, roma_indoor
from romatch.utils.utils import tensor_to_pil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up device
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

logger.info(f"Using device: {device}")


def load_json_config(json_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        json_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {json_path}")
        return config
    except FileNotFoundError:
        logger.error(f"JSON configuration file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {json_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON configuration: {e}")
        raise


def merge_configs(json_config: Dict, cli_args: Dict) -> Dict:
    """
    Merge JSON configuration with command line arguments.
    Command line arguments take precedence over JSON config.
    
    Args:
        json_config: Configuration loaded from JSON file
        cli_args: Command line arguments (only non-None values)
        
    Returns:
        Merged configuration dictionary
    """
    # Start with JSON config as base
    merged_config = json_config.copy()
    
    # Override with command line arguments (only if they were explicitly provided)
    for key, value in cli_args.items():
        if value is not None:
            merged_config[key] = value
            logger.info(f"CLI override: {key} = {value}")
    
    return merged_config


def save_config_to_output(config: Dict, output_dir: str) -> str:
    """
    Save the final configuration to output directory.
    
    Args:
        config: Configuration dictionary to save
        output_dir: Output directory path
        
    Returns:
        Path to saved configuration file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_file = output_path / "run_config.json"
    
    # Create a clean config for saving (remove any non-serializable items)
    clean_config = {}
    for key, value in config.items():
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            clean_config[key] = value
        except (TypeError, ValueError):
            # Convert to string if not directly serializable
            clean_config[key] = str(value)
    
    with open(config_file, 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    logger.info(f"Saved run configuration to: {config_file}")
    return str(config_file)


def create_example_config(output_path: str):
    """
    Create an example configuration file with all available options.
    
    Args:
        output_path: Path where to save the example config
    """
    example_config = {
        "scene_dir": "/path/to/your/scene",
        "colmap_path": None,
        "images_dir": None,
        "output_dir": None,
        "model_type": "tiny_roma",
        "resolution": [864, 1152],
        "min_common_points": 100,
        "min_baseline": 0.1,
        "max_baseline": 2.0,
        "max_pairs_per_image": 5,
        "max_points": 100000,
        "pairs_file": "pairs.json",
        "disable_bbox_filter": False,
        "min_triangulation_angle": 2.0,
        "enable_visualizations": False,

        "cache_size": 100
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    logger.info(f"Created example configuration file: {output_path}")


class Profiler:
    """Simple profiler to track time spent in different operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager to time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[operation_name].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'count': len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return summary
    
    def print_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_summary()
        if not summary:
            logger.info("No profiling data available")
            return
        
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE PROFILING SUMMARY")
        logger.info("="*60)
        
        # Sort by total time (descending)
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for operation, stats in sorted_ops:
            logger.info(f"\n{operation}:")
            logger.info(f"  Total time: {stats['total_time']:.3f}s")
            logger.info(f"  Average time: {stats['average_time']:.3f}s")
            logger.info(f"  Count: {stats['count']}")
            if stats['count'] > 1:
                logger.info(f"  Min/Max: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
        
        logger.info("="*60)
    
    def save_to_file(self, filepath: str):
        """Save profiling results to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


class ImageCache:
    """Smart image cache to avoid repeated loading and resizing."""
    
    def __init__(self, max_cache_size: int = 50):
        """
        Initialize image cache.
        
        Args:
            max_cache_size: Maximum number of cached image variants
        """
        self.max_cache_size = max_cache_size
        self.cache = {}  # Key: (image_path, width, height, is_numpy), Value: cached_image
        self.access_order = []  # For LRU eviction
        
    def _make_key(self, image_path: str, width: int = None, height: int = None, as_numpy: bool = False):
        """Create cache key."""
        return (str(image_path), width, height, as_numpy)
    
    def _evict_lru(self):
        """Remove least recently used item."""
        if len(self.cache) >= self.max_cache_size and self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]
    
    def _update_access(self, key):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_image(self, image_path: str, width: int = None, height: int = None, as_numpy: bool = False):
        """
        Get image from cache or load it.
        
        Args:
            image_path: Path to image file
            width, height: Target size (None for original size)
            as_numpy: Return as numpy array (0-1 float) instead of PIL Image
            
        Returns:
            PIL Image or numpy array
        """
        key = self._make_key(image_path, width, height, as_numpy)
        
        # Check cache
        if key in self.cache:
            self._update_access(key)
            return self.cache[key]
        
        # Load and potentially resize image
        img = Image.open(image_path)
        
        if width is not None and height is not None:
            img = img.resize((width, height), Image.LANCZOS)
        
        # Convert to numpy if requested
        if as_numpy:
            result = np.array(img) / 255.0
        else:
            result = img
        
        # Cache the result
        self._evict_lru()
        self.cache[key] = result
        self._update_access(key)
        
        return result

    def clear(self):
        """Clear all cached images."""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self):
        """Get cache statistics."""
        total_memory = 0
        for img in self.cache.values():
            if hasattr(img, 'nbytes'):  # numpy array
                total_memory += img.nbytes
            elif hasattr(img, 'size'):  # PIL Image  
                # Estimate PIL image memory usage
                width, height = img.size
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                total_memory += width * height * channels
        
        return {
            'cached_items': len(self.cache),
            'max_size': self.max_cache_size,
            'memory_usage_mb': total_memory / (1024*1024)
        }


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Efficiently get image dimensions from file headers without loading the full image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        (width, height) tuple
    """
    with Image.open(image_path) as img:
        return img.size


def get_image_for_model(image_path: str, 
                       model_type: str,
                       image_cache: 'ImageCache',
                       target_size: Tuple[int, int] = None,
                       as_numpy: bool = False) -> Image.Image:
    """
    Get image loaded and sized appropriately for the specified model type.
    Handles caching for efficient image loading and resizing.
    
    Args:
        image_path: Path to original image file
        model_type: "roma_outdoor", "roma_indoor", or "tiny_roma"
        image_cache: ImageCache instance for caching
        target_size: (width, height) for final image size
        as_numpy: Return as numpy array instead of PIL Image
        
    Returns:
        PIL Image or numpy array with the image data
    """
    if model_type in ["roma_outdoor", "roma_indoor"]:
        # For roma models, use the specified target size and cache
        if target_size is None:
            # No target size specified, load at original resolution
            return image_cache.get_image(image_path, as_numpy=as_numpy)
        else:
            width, height = target_size
            return image_cache.get_image(image_path, width, height, as_numpy=as_numpy)
    
    elif model_type == "tiny_roma":
        # For tiny_roma, use original images with caching
        if target_size is None:
            return image_cache.get_image(image_path, as_numpy=as_numpy)
        else:
            width, height = target_size
            return image_cache.get_image(image_path, width, height, as_numpy=as_numpy)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")





def generate_depth_map_from_camera_coordinates(points_3d_cam1: np.ndarray,
                                              kpts1: np.ndarray,
                                              image_width: int,
                                              image_height: int) -> np.ndarray:
    """
    Generate a depth map for image 1 from 3D points already in camera 1's coordinate system.
    Uses vectorized operations for fast computation.
    
    Args:
        points_3d_cam1: 3D points in camera 1's coordinate system (N, 3)
        kpts1: 2D keypoints in image 1 (N, 2) - integer pixel locations at warp resolution
        image_width, image_height: Current image resolution (warp resolution)
        
    Returns:
        depth_map: (H, W) depth map with depths at keypoint locations, zeros elsewhere
    """
    
    # Extract depths (Z coordinates in camera coordinate system)
    depths = points_3d_cam1[:, 2]
    
    # Filter out points behind the camera (negative depth)
    valid_mask = depths > 0
    if not np.any(valid_mask):
        # No valid points, return empty depth map
        return np.zeros((image_height, image_width), dtype=np.float32)
    
    # Keep only valid points
    valid_kpts1 = kpts1[valid_mask]
    valid_depths = depths[valid_mask]
    
    # Convert keypoints to integer pixel coordinates
    # kpts1 should already be integer locations, but ensure they are
    x_coords = np.round(valid_kpts1[:, 0]).astype(np.int32)
    y_coords = np.round(valid_kpts1[:, 1]).astype(np.int32)
    
    # Filter points that are within image bounds
    bounds_mask = (
        (x_coords >= 0) & (x_coords < image_width) &
        (y_coords >= 0) & (y_coords < image_height)
    )
    
    if not np.any(bounds_mask):
        # No points within bounds
        return np.zeros((image_height, image_width), dtype=np.float32)
    
    # Keep only in-bounds points
    x_coords = x_coords[bounds_mask]
    y_coords = y_coords[bounds_mask]
    final_depths = valid_depths[bounds_mask]
    
    # Initialize depth map
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Directly assign depths to corresponding pixel locations
    # Since each 3D point came from a unique kpts1 location, no conflicts possible
    depth_map[y_coords, x_coords] = final_depths
    
    return depth_map


class DenseMatchingPipeline:
    """Pipeline for dense matching and point cloud generation."""
    
    def __init__(self, 
                 colmap_path: str,
                 images_dir: str,
                 output_dir: str,
                 model_type: str = "tiny_roma",
                 resolution: Tuple[int, int] = (864, 1152),
                 min_triangulation_angle: float = 2.0,
                 save_visualizations: bool = False,
                 command_args: Optional[Dict] = None,
                 cache_size: int = 100,
                 include_cameras_in_bbox: bool = False,
                 pair_bbox_min_track_size: int = 3,
                 pair_bbox_margin: float = 0.1):
        """
        Initialize dense matching pipeline.
        
        Args:
            colmap_path: Path to COLMAP sparse reconstruction
            images_dir: Directory containing images
            output_dir: Output directory for results
            model_type: RoMa model type ("tiny_roma", "roma_outdoor", or "roma_indoor")
            resolution: Target resolution for matching
            min_triangulation_angle: Minimum triangulation angle in degrees
            save_visualizations: Whether to save match visualizations (default: False)

            command_args: Dictionary of command line arguments used to call the program
            cache_size: Maximum number of cached image variants (higher = faster but more memory)
            include_cameras_in_bbox: Whether to include camera centers in scene bounding box computation
            pair_bbox_min_track_size: Minimum track size for points used in pair bounding box computation
            pair_bbox_margin: Margin factor for pair bounding box as fraction of box size
        """
        self.colmap_path = Path(colmap_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.min_triangulation_angle = min_triangulation_angle
        self.save_visualizations = save_visualizations
        self.include_cameras_in_bbox = include_cameras_in_bbox
        self.pair_bbox_min_track_size = pair_bbox_min_track_size
        self.pair_bbox_margin = pair_bbox_margin
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "point_clouds").mkdir(exist_ok=True)
        (self.output_dir / "matches").mkdir(exist_ok=True)
        if self.save_visualizations:
            (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Load COLMAP data
        logger.info("Loading COLMAP reconstruction...")
        self.colmap_dataset = COLMAPDataset(str(self.colmap_path))
        
        # Initialize RoMa model
        logger.info(f"Initializing {model_type} model...")
        if model_type == "roma_outdoor":
            self.roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=resolution)
        elif model_type == "roma_indoor":
            self.roma_model = roma_indoor(device=device, coarse_res=560, upsample_res=resolution)
        elif model_type == "tiny_roma":
            self.roma_model = tiny_roma_v1_outdoor(device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        
        # Store the actual resolution used by the model
        if model_type in ["roma_outdoor", "roma_indoor"]:
            self.actual_resolution = self.roma_model.get_output_resolution()  # (H, W)
        else:  # tiny_roma - resolution will be determined during first match
            self.actual_resolution = None
        
        # Store command line arguments for summary
        self.command_args = command_args or {}
        
        # Storage for point clouds
        self.point_clouds = []
        
        # Storage for point cloud metadata (for summary)
        self.point_cloud_metadata = []  # List of (img_id1, img_id2, point_count)
        
        # Performance profiler
        self.profiler = Profiler()
        
        # Image cache for faster loading
        self.image_cache = ImageCache(max_cache_size=cache_size)
        
        # Scene bounding box (computed once)
        self.scene_bbox = None
        

    

        
    def select_image_pairs(self, 
                          min_common_points: int = 100,
                          min_baseline: float = 0.1,
                          max_baseline: float = 2.0,
                          max_pairs_per_image: int = 5,
                          save_pairs_file: Optional[str] = None) -> List[Tuple[int, int, Dict]]:
        """Select good image pairs for dense matching using per-image selection."""
        logger.info("Selecting image pairs...")
        pairs = self.colmap_dataset.select_image_pairs(
            min_common_points=min_common_points,
            min_baseline=min_baseline, 
            max_baseline=max_baseline,
            max_pairs_per_image=max_pairs_per_image,
            save_pairs_file=save_pairs_file
        )
        
        logger.info(f"Selected {len(pairs)} image pairs")
        for i, (id1, id2, meta) in enumerate(pairs[:5]):  # Show top 5
            img1_name = self.colmap_dataset.images[id1].name
            img2_name = self.colmap_dataset.images[id2].name
            logger.info(f"  {i+1}. {img1_name} <-> {img2_name} "
                       f"(baseline: {meta['baseline']:.3f}, "
                       f"common_points: {meta['common_points']}, "
                       f"quality: {meta['quality']:.1f})")
        
        return pairs
    
    def compute_scene_bounding_box(self, 
                                  min_visibility: int = 3,
                                  include_cameras: bool = None,
                                  margin_factor: float = 0.15) -> Dict:
        """Compute scene bounding box from COLMAP data."""
        if self.scene_bbox is None:
            # Use instance variable if include_cameras not explicitly set
            if include_cameras is None:
                include_cameras = self.include_cameras_in_bbox
                
            logger.info(f"Computing scene bounding box for point cloud filtering (include_cameras={include_cameras})...")
            try:
                self.scene_bbox = self.colmap_dataset.compute_scene_bounding_box(
                    min_visibility=min_visibility,
                    include_cameras=include_cameras,
                    margin_factor=margin_factor,
                    robust_percentile=95.0
                )
                
                # Save bounding box visualization
                bbox_path = self.output_dir / "scene_bounding_box.ply"
                self.colmap_dataset.export_bbox_to_ply(self.scene_bbox, str(bbox_path))
                logger.info(f"Saved scene bounding box to {bbox_path}")
                
            except Exception as e:
                logger.warning(f"Failed to compute scene bounding box: {e}")
                self.scene_bbox = None
        
        return self.scene_bbox
    
    def dense_match_pair(self, img_id1: int, img_id2: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Perform dense matching between two images.
        
        Returns:
            warp: Dense correspondence field
            certainty: Matching certainty map
            info: Metadata about the matching
        """
        # Get image paths
        img_path1 = self.colmap_dataset.get_image_path(img_id1, self.images_dir)
        img_path2 = self.colmap_dataset.get_image_path(img_id2, self.images_dir)
        
        logger.info(f"Dense matching: {img_path1.name} <-> {img_path2.name}")
        
        # Clear GPU cache before processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load and resize images using the unified function
        start_time = time.time()
        
        if self.model_type in ["roma_outdoor", "roma_indoor"]:
            with self.profiler.profile("image_loading_and_resize"):
                H, W = self.roma_model.get_output_resolution()
                # Use unified image loading function
                img1 = get_image_for_model(str(img_path1), self.model_type, self.image_cache, target_size=(W, H))
                img2 = get_image_for_model(str(img_path2), self.model_type, self.image_cache, target_size=(W, H))
            
            # Perform matching
            with self.profiler.profile("roma_model_inference"):
                warp, certainty = self.roma_model.match(str(img_path1), str(img_path2), device=device)
            
        else:  # tiny_roma
            # Resize on the fly using unified function
            max_size = 800  # Limit maximum dimension to control memory usage
            
            with self.profiler.profile("image_loading_and_resize"):
                # Get original image dimensions efficiently
                orig_w1, orig_h1 = get_image_dimensions(str(img_path1))
                orig_w2, orig_h2 = get_image_dimensions(str(img_path2))
                
                # Calculate target sizes while maintaining aspect ratio
                def get_resized_dimensions(w, h, max_size):
                    if max(w, h) > max_size:
                        if w > h:
                            return max_size, int(h * max_size / w)
                        else:
                            return int(w * max_size / h), max_size
                    return w, h
                
                new_w1, new_h1 = get_resized_dimensions(orig_w1, orig_h1, max_size)
                new_w2, new_h2 = get_resized_dimensions(orig_w2, orig_h2, max_size)
                
                # Get resized images using unified function
                img1_resized = get_image_for_model(str(img_path1), self.model_type, self.image_cache, 
                                                 target_size=(new_w1, new_h1))
                img2_resized = get_image_for_model(str(img_path2), self.model_type, self.image_cache, 
                                                 target_size=(new_w2, new_h2))
            
            # Match directly with PIL Images - no temporary files needed!
            with self.profiler.profile("tiny_roma_model_inference"):
                warp, certainty = self.roma_model.match(img1_resized, img2_resized)
            
            H, W = warp.shape[:2]
            # Store actual resolution for tiny_roma on first match
            if self.actual_resolution is None:
                self.actual_resolution = (H, W)
            logger.info(f"  Warp shape: {warp.shape}")
            logger.info(f"  Certainty shape: {certainty.shape}")
            logger.info(f"  Resized image 1: {img1_resized.size}")
            logger.info(f"  Resized image 2: {img2_resized.size}")
        
        match_time = time.time() - start_time
        
        # Calculate statistics
        mean_certainty = certainty.mean().item()
        max_certainty = certainty.max().item()
        high_conf_ratio = (certainty > 0.5).float().mean().item()
        
        info = {
            'img_id1': img_id1,
            'img_id2': img_id2,
            'img_name1': img_path1.name,
            'img_name2': img_path2.name,
            'resolution': (H, W),
            'match_time': match_time,
            'mean_certainty': mean_certainty,
            'max_certainty': max_certainty,
            'high_conf_ratio': high_conf_ratio,
            'total_pixels': H * W
        }
        
        logger.info(f"  Match time: {match_time:.2f}s, "
                   f"Mean certainty: {mean_certainty:.3f}, "
                   f"High conf ratio: {high_conf_ratio:.3f}")
        
        return warp, certainty, info
    
    def _create_image_pyramid(self, image_path: str, pyramid_levels: int) -> list[tuple[np.ndarray, int, int]]:
        """
        Create image pyramid with specified number of levels.
        
        Args:
            image_path: Path to the image
            pyramid_levels: Number of pyramid levels
            
        Returns:
            List of tuples (image_array, width, height) for each pyramid level
        """
        pyramid = []
        
        # Load original image
        orig_image = Image.open(image_path)
        orig_w, orig_h = orig_image.size
        
        for level in range(pyramid_levels):
            # Calculate dimensions for this level
            scale_factor = 2 ** level
            level_w = max(64, orig_w // scale_factor)  # Minimum width
            level_h = max(64, orig_h // scale_factor)  # Minimum height
            
            # Resize image
            if level == 0:
                # Original resolution
                level_image = orig_image
            else:
                level_image = orig_image.resize((level_w, level_h), Image.LANCZOS)
            
            # Convert to numpy array
            level_image_np = np.array(level_image)
            if len(level_image_np.shape) == 3:
                level_image_np = level_image_np.transpose(2, 0, 1)  # HWC to CHW
            else:
                level_image_np = level_image_np[None, :]  # Add channel dimension for grayscale
                
            pyramid.append((level_image_np, level_w, level_h))
            
            logger.debug(f"  Pyramid level {level}: {level_w}x{level_h} (scale: 1/{scale_factor})")
        
        return pyramid
    
    def _scale_intrinsics(self, K: np.ndarray, original_size: tuple[int, int], 
                         target_size: tuple[int, int]) -> np.ndarray:
        """
        Scale camera intrinsics matrix for different image resolutions.
        
        Args:
            K: Original camera intrinsics matrix (3x3)
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
            
        Returns:
            Scaled intrinsics matrix
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        # Calculate scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Scale the intrinsics
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        
        return K_scaled
    
    def _merge_point_clouds(self, point_clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """
        Merge multiple point clouds from different resolution levels.
        
        Args:
            point_clouds: List of point clouds from different pyramid levels
            
        Returns:
            Merged point cloud
        """
        if not point_clouds:
            return o3d.geometry.PointCloud()
        
        # Start with the first cloud
        merged_cloud = o3d.geometry.PointCloud()
        all_points = []
        all_colors = []
        
        for i, pcd in enumerate(point_clouds):
            if len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                all_points.append(points)
                
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    all_colors.append(colors)
                else:
                    # Default color for points without color
                    default_color = np.array([0.7, 0.7, 0.7])  # Gray
                    colors = np.tile(default_color, (len(points), 1))
                    all_colors.append(colors)
        
        if all_points:
            # Combine all points and colors
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors) if all_colors else None
            
            # Create merged point cloud
            merged_cloud.points = o3d.utility.Vector3dVector(merged_points)
            if merged_colors is not None:
                merged_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
            
            # Remove duplicate points (points that are very close to each other)
            merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.01)
            
            logger.info(f"  Merged {len(point_clouds)} pyramid levels into {len(merged_cloud.points)} points")
        
        return merged_cloud
    
    def _grid_sample_points(self, valid_indices: torch.Tensor, certainty: torch.Tensor, 
                           base_threshold: float, max_points: int, grid_percentage: float,
                           max_threshold_multiplier: float, H: int, W: int) -> torch.Tensor:
        """
        Sample points using a grid-based approach for more even distribution.
        
        Args:
            valid_indices: Tensor of valid point coordinates (N, 2) where N is number of valid points
            certainty: Full certainty map (H, W)
            base_threshold: Base certainty threshold
            max_points: Maximum number of points to sample
            grid_percentage: Percentage for grid cell size (e.g., 10.0 means 10% of image dimension)
            max_threshold_multiplier: Maximum multiplier for threshold when cells have insufficient points
            H, W: Image height and width
            
        Returns:
            Sampled valid_indices tensor
        """
        # Calculate grid dimensions
        grid_size_h = max(1, int(H * grid_percentage / 100.0))
        grid_size_w = max(1, int(W * grid_percentage / 100.0))
        n_cells_h = (H + grid_size_h - 1) // grid_size_h  # Ceiling division
        n_cells_w = (W + grid_size_w - 1) // grid_size_w
        total_cells = n_cells_h * n_cells_w
        
        # Target points per cell
        points_per_cell = max_points // total_cells
        remainder_points = max_points % total_cells
        
        logger.info(f"  Grid sampling: {n_cells_h}x{n_cells_w} grid ({grid_size_h}x{grid_size_w} pixels per cell)")
        logger.info(f"  Target: ~{points_per_cell} points per cell, {remainder_points} remainder")
        
        selected_indices = []
        
        for i in range(n_cells_h):
            for j in range(n_cells_w):
                # Define cell boundaries
                y_start = i * grid_size_h
                y_end = min((i + 1) * grid_size_h, H)
                x_start = j * grid_size_w
                x_end = min((j + 1) * grid_size_w, W)
                
                # Target points for this cell (distribute remainder across first cells)
                cell_target = points_per_cell + (1 if (i * n_cells_w + j) < remainder_points else 0)
                
                if cell_target == 0:
                    continue
                
                # Find valid indices in this cell
                cell_mask = ((valid_indices[:, 0] >= y_start) & (valid_indices[:, 0] < y_end) &
                            (valid_indices[:, 1] >= x_start) & (valid_indices[:, 1] < x_end))
                cell_indices = valid_indices[cell_mask]
                
                # If we have enough points, sample them
                if len(cell_indices) >= cell_target:
                    perm = torch.randperm(len(cell_indices))[:cell_target]
                    selected_indices.append(cell_indices[perm])
                else:
                    # Not enough points - try increasing threshold up to max_threshold_multiplier
                    current_threshold = base_threshold
                    threshold_multiplier = 1.0
                    
                    while (len(cell_indices) < cell_target and 
                           threshold_multiplier <= max_threshold_multiplier):
                        # Decrease threshold to get more points
                        threshold_multiplier *= 1.5  # Increase by 50% each iteration
                        current_threshold = base_threshold / threshold_multiplier
                        
                        # Create new mask for this cell with lower threshold
                        cell_certainty = certainty[y_start:y_end, x_start:x_end]
                        cell_valid_mask = cell_certainty > current_threshold
                        cell_y_coords, cell_x_coords = torch.nonzero(cell_valid_mask, as_tuple=True)
                        
                        # Convert to global coordinates
                        global_y = cell_y_coords + y_start
                        global_x = cell_x_coords + x_start
                        cell_indices = torch.stack([global_y, global_x], dim=1)
                    
                    # Sample what we have (might be less than target)
                    if len(cell_indices) > 0:
                        actual_sample = min(len(cell_indices), cell_target)
                        if len(cell_indices) > actual_sample:
                            perm = torch.randperm(len(cell_indices))[:actual_sample]
                            selected_indices.append(cell_indices[perm])
                        else:
                            selected_indices.append(cell_indices)
        
        # Combine all selected indices
        if selected_indices:
            result = torch.cat(selected_indices, dim=0)
            logger.info(f"  Grid sampling selected {len(result)} points from {len(valid_indices)} candidates")
            return result
        else:
            # Fallback to random sampling if grid sampling failed
            logger.warning(f"  Grid sampling failed, falling back to random sampling")
            perm = torch.randperm(len(valid_indices))[:max_points]
            return valid_indices[perm]
    
    def generate_point_cloud(self,
                           warp: torch.Tensor,
                           certainty: torch.Tensor,
                           img_id1: int,
                           img_id2: int,
                           certainty_threshold: float = 0.3,
                           max_points: int = 100000,
                           use_bbox_filter: bool = True,
                           generate_depth_map: bool = True,
                           use_grid_sampling: bool = False,
                           grid_percentage: float = 10.0,
                           max_threshold_multiplier: float = 5.0) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Generate point cloud from dense matching results.
        
        Args:
            warp: Dense correspondence field
            certainty: Matching certainty map
            img_id1, img_id2: Image IDs
            certainty_threshold: Minimum certainty for triangulation
            max_points: Maximum number of points to triangulate
            use_bbox_filter: Whether to filter points using scene bounding box
            generate_depth_map: Whether to generate depth map for image 1
            use_grid_sampling: Whether to use grid-based sampling instead of random sampling
            grid_percentage: Percentage of image to use for each grid cell (e.g., 10.0 means 10% => 10x10 grid)
            max_threshold_multiplier: Maximum multiplier for certainty_threshold when grid cells have insufficient points
            
        Returns:
            tuple: (point_cloud, depth_map) where depth_map is None if generate_depth_map=False
        """
        logger.info(f"Generating point cloud for pair {img_id1}-{img_id2}")
        
        # Get camera information
        with self.profiler.profile("camera_info_retrieval"):
            info1 = self.colmap_dataset.get_image_info(img_id1)
            info2 = self.colmap_dataset.get_image_info(img_id2)
            
            K1, K2 = info1['K'], info2['K']
            R_rel, t_rel = self.colmap_dataset.get_relative_pose(img_id1, img_id2)
        
        # Get camera centers for verification
        cam_center1 = self.colmap_dataset.images[img_id1].projection_center()
        cam_center2 = self.colmap_dataset.images[img_id2].projection_center()
        baseline_actual = np.linalg.norm(cam_center2 - cam_center1)
        
        logger.info(f"  Camera centers: {cam_center1} -> {cam_center2}")
        logger.info(f"  Baseline: {baseline_actual:.3f}m")
        
        # Load images for colors (with smart caching for tiny_roma)
        with self.profiler.profile("image_loading_for_colors"):
            img_path1 = self.colmap_dataset.get_image_path(img_id1, self.images_dir)
            img_path2 = self.colmap_dataset.get_image_path(img_id2, self.images_dir)
            
            # Get warp resolution 
            H, W = warp.shape[:2]
            
            # Use original images - cache handles resizing efficiently
            orig_w1, orig_h1 = get_image_dimensions(str(img_path1))
            orig_w2, orig_h2 = get_image_dimensions(str(img_path2))
            img1_np = self.image_cache.get_image(str(img_path1), W, H, as_numpy=True)
            img2_np = self.image_cache.get_image(str(img_path2), W, H, as_numpy=True)
        
        # Sample high-certainty correspondences
        with self.profiler.profile("correspondence_extraction"):
            mask = certainty > certainty_threshold
            valid_indices = torch.nonzero(mask, as_tuple=False)
            
            if len(valid_indices) > max_points:
                if use_grid_sampling:
                    # Grid-based sampling for more even distribution
                    valid_indices = self._grid_sample_points(
                        valid_indices, certainty, certainty_threshold, 
                        max_points, grid_percentage, max_threshold_multiplier, H, W
                    )
                else:
                    # Randomly sample points (original behavior)
                    perm = torch.randperm(len(valid_indices))[:max_points]
                    valid_indices = valid_indices[perm]
            
            logger.info(f"  Triangulating {len(valid_indices)} points")
            
            # Extract correspondences
            y_coords, x_coords = valid_indices[:, 0], valid_indices[:, 1]
            
            # Points in image 1 (pixel coordinates)
            kpts1 = torch.stack([x_coords, y_coords], dim=1).float()
            
            # Corresponding points in image 2 from warp
            warp_coords = warp[y_coords, x_coords, 2:4]  # Get the second image coordinates
            
            # Convert from normalized coordinates to pixel coordinates
            kpts2_x = W * (warp_coords[:, 0] + 1) / 2
            kpts2_y = H * (warp_coords[:, 1] + 1) / 2
            kpts2 = torch.stack([kpts2_x, kpts2_y], dim=1)
            
            # Convert to numpy
            kpts1_np = kpts1.cpu().numpy()
            kpts2_np = kpts2.cpu().numpy()
        
        # Adjust camera matrices for resized images
        with self.profiler.profile("camera_matrix_scaling"):
            # Scale factors from original image size to warp resolution
            scale_x1, scale_y1 = W / orig_w1, H / orig_h1
            scale_x2, scale_y2 = W / orig_w2, H / orig_h2
            
            K1_scaled = K1.copy()
            K1_scaled[0, 0] *= scale_x1  # fx
            K1_scaled[1, 1] *= scale_y1  # fy
            K1_scaled[0, 2] *= scale_x1  # cx
            K1_scaled[1, 2] *= scale_y1  # cy
            
            K2_scaled = K2.copy()
            K2_scaled[0, 0] *= scale_x2  # fx
            K2_scaled[1, 1] *= scale_y2  # fy
            K2_scaled[0, 2] *= scale_x2  # cx
            K2_scaled[1, 2] *= scale_y2  # cy
        
        # Triangulate 3D points
        with self.profiler.profile("triangulation"):
            points_3d_cam1 = triangulate_points(kpts1_np, kpts2_np, K1_scaled, K2_scaled, R_rel, t_rel)
        
        # Transform points from camera 1 coordinate system to world coordinates
        with self.profiler.profile("coordinate_transformation"):
            R1, t1 = self.colmap_dataset.get_pose(img_id1)
            # COLMAP uses cam_from_world: X_cam = R @ X_world + t
            # So to transform from camera to world: X_world = R^T @ (X_cam - t)
            t1_flat = t1.flatten()  # Ensure t1 is 1D for broadcasting
            points_3d_world = (R1.T @ (points_3d_cam1 - t1_flat).T).T
            
            # Use world coordinates for further processing
            points_3d = points_3d_world
        
        logger.info(f"  Point cloud center: [{np.mean(points_3d, axis=0)[0]:.3f}, {np.mean(points_3d, axis=0)[1]:.3f}, {np.mean(points_3d, axis=0)[2]:.3f}]")
        
        # Generate depth map for image 1 if requested
        depth_map = None
        if generate_depth_map:
            with self.profiler.profile("depth_map_generation"):
                depth_map = generate_depth_map_from_camera_coordinates(
                    points_3d_cam1, kpts1_np, W, H
                )
                logger.info(f"  Generated depth map: {depth_map.shape}, "
                           f"non-zero pixels: {np.count_nonzero(depth_map)}, "
                           f"depth range: [{np.min(depth_map[depth_map > 0]):.3f}, {np.max(depth_map):.3f}]")
        
        # Get colors from image 1
        with self.profiler.profile("color_extraction"):
            colors = img1_np[y_coords.cpu().numpy(), x_coords.cpu().numpy()]
        
        # Filter out points that are too close or too far (using world coordinates)
        with self.profiler.profile("depth_filtering"):
            depths = np.linalg.norm(points_3d, axis=1)
            valid_depth_mask = (depths > 0.1) & (depths < 100)
            
            points_3d = points_3d[valid_depth_mask]
            colors = colors[valid_depth_mask]
            
            logger.info(f"  After depth filtering: {len(points_3d)} points")
        
        # Filter points by triangulation angle (remove points with small viewing angles)
        with self.profiler.profile("triangulation_angle_filtering"):
            points_3d, colors = self.filter_by_triangulation_angle(
                points_3d, colors, cam_center1, cam_center2, min_angle_degrees=self.min_triangulation_angle
            )
        
        # Optional: Filter by pair-specific bounding box
        if use_bbox_filter and len(points_3d) > 0:
            with self.profiler.profile("pair_bounding_box_filtering"):
                try:
                    pair_bbox = self.colmap_dataset.compute_pair_bounding_box(
                        img_id1, img_id2, 
                        min_track_size=self.pair_bbox_min_track_size, 
                        margin_factor=self.pair_bbox_margin
                    )
                    
                    # Filter points using pair-specific bounding box
                    bbox_min, bbox_max = pair_bbox['min'], pair_bbox['max']
                    bbox_filter = (
                        (points_3d[:, 0] >= bbox_min[0]) & (points_3d[:, 0] <= bbox_max[0]) &
                        (points_3d[:, 1] >= bbox_min[1]) & (points_3d[:, 1] <= bbox_max[1]) &
                        (points_3d[:, 2] >= bbox_min[2]) & (points_3d[:, 2] <= bbox_max[2])
                    )
                    
                    filtered_points = points_3d[bbox_filter]
                    filtered_colors = colors[bbox_filter]
                    
                    logger.info(f"  Pair bbox filter ({pair_bbox['num_points']} ref points): {len(filtered_points)}/{len(points_3d)} points retained")
                    points_3d = filtered_points
                    colors = filtered_colors
                    
                except ValueError as e:
                    logger.warning(f"  Pair bbox filtering failed: {e}, skipping bbox filter")
                except Exception as e:
                    logger.warning(f"  Pair bbox filtering error: {e}, skipping bbox filter")
        
        logger.info(f"  Final point cloud: {len(points_3d)} points")
        
        # Create Open3D point cloud
        with self.profiler.profile("point_cloud_creation"):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove statistical outliers
        with self.profiler.profile("outlier_removal"):
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd, depth_map
    
    def save_depth_map(self, depth_map: np.ndarray, img_id: int, output_path: str = None):
        """
        Save depth map as both raw numpy array and visualization.
        
        Args:
            depth_map: (H, W) depth map
            img_id: Image ID for filename
            output_path: Optional custom output path
        """
        if depth_map is None:
            logger.warning("No depth map to save")
            return
            
        if output_path is None:
            img_name = self.colmap_dataset.images[img_id].name
            base_name = Path(img_name).stem
            output_path = self.output_dir / "depth_maps" / f"{base_name}_depth.npz"
        else:
            output_path = Path(output_path)
            
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save raw depth map
        np.savez_compressed(output_path, depth=depth_map)
        logger.info(f"Saved depth map to {output_path}")
        
        # Create and save visualization
        vis_path = output_path.with_suffix('.png')
        try:
            self._save_depth_visualization(depth_map, vis_path)
            logger.info(f"Saved depth visualization to {vis_path}")
        except Exception as e:
            logger.warning(f"Failed to save depth visualization: {e}")
            logger.info("Depth map data saved successfully, but visualization skipped")
    
    def _save_depth_visualization(self, depth_map: np.ndarray, vis_path: Path):
        """Create and save depth map visualization using OpenCV."""
        import cv2
        
        # Mask for valid depths
        valid_mask = depth_map > 0
        
        if not np.any(valid_mask):
            logger.warning("No valid depths to visualize")
            return
        
        # Normalize depth values for visualization
        min_depth = np.min(depth_map[valid_mask])
        max_depth = np.max(depth_map[valid_mask])
        
        # Create normalized depth map (0-255 range)
        norm_depth = np.zeros_like(depth_map, dtype=np.uint8)
        if max_depth > min_depth:
            norm_depth[valid_mask] = ((depth_map[valid_mask] - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        
        # Apply viridis colormap using OpenCV
        colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_VIRIDIS)
        # Set invalid pixels to black
        colored_depth[~valid_mask] = [0, 0, 0]
        
        # Add text overlay with depth info
        text_color = (255, 255, 255)  # White text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text1 = f'Depth Range: [{min_depth:.3f}, {max_depth:.3f}]m'
        text2 = f'Valid pixels: {np.count_nonzero(valid_mask)}/{depth_map.size}'
        
        cv2.putText(colored_depth, text1, (10, 30), font, font_scale, text_color, thickness)
        cv2.putText(colored_depth, text2, (10, 60), font, font_scale, text_color, thickness)
        
        # Save image (OpenCV uses BGR format)
        cv2.imwrite(str(vis_path), colored_depth)
    
    def save_visualization(self, 
                          warp: torch.Tensor,
                          certainty: torch.Tensor,
                          img_id1: int,
                          img_id2: int):
        """Save matching visualization."""
        img_path1 = self.colmap_dataset.get_image_path(img_id1, self.images_dir)
        img_path2 = self.colmap_dataset.get_image_path(img_id2, self.images_dir)
        
        H, W = warp.shape[:2]
        
        # Load and resize images using unified function
        img1 = get_image_for_model(str(img_path1), self.model_type, self.image_cache, target_size=(W, H))
        img2 = get_image_for_model(str(img_path2), self.model_type, self.image_cache, target_size=(W, H))
        
        # Convert to tensors
        x1 = (torch.tensor(np.array(img1)) / 255).to(device).permute(2, 0, 1)
        x2 = (torch.tensor(np.array(img2)) / 255).to(device).permute(2, 0, 1)
        
        # Create warped visualization
        if self.model_type in ["roma_outdoor", "roma_indoor"]:
            im2_transfer_rgb = F.grid_sample(
                x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
            )[0]
            im1_transfer_rgb = F.grid_sample(
                x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
            
            # Create certainty overlay
            white_im = torch.ones((H, warp_im.shape[2]), device=device)
            vis_im = certainty * warp_im + (1 - certainty) * white_im
        else:
            # For tiny RoMa, create side-by-side visualization
            warp_im = torch.cat((x1, x2), dim=2)
            
            # Create certainty overlay with duplicated certainty for side-by-side
            certainty_expanded = torch.cat([certainty, certainty], dim=1)  # Duplicate horizontally
            white_im = torch.ones((H, warp_im.shape[2]), device=device)
            vis_im = certainty_expanded * warp_im + (1 - certainty_expanded) * white_im
        
        # Save visualization
        vis_path = self.output_dir / "visualizations" / f"match_{img_id1}_{img_id2}.jpg"
        tensor_to_pil(vis_im, unnormalize=False).save(vis_path)
        
        # Save certainty map
        cert_path = self.output_dir / "visualizations" / f"certainty_{img_id1}_{img_id2}.jpg"
        certainty_vis = torch.stack([certainty, certainty, certainty], dim=0)
        tensor_to_pil(certainty_vis, unnormalize=False).save(cert_path)
        
        logger.info(f"  Saved visualizations: {vis_path.name}, {cert_path.name}")
    
    def _process_pair_multires(self, img_id1: int, img_id2: int, pair_idx: int, 
                              use_bbox_filter: bool = True, max_points: int = 100000,
                              use_grid_sampling: bool = False, grid_percentage: float = 10.0, 
                              max_threshold_multiplier: float = 5.0, pyramid_levels: int = 3,
                              debug: bool = False) -> o3d.geometry.PointCloud:
        """
        Process a single image pair using multi-resolution approach.
        
        Args:
            img_id1, img_id2: Image IDs to process
            pair_idx: Index of this pair for logging
            use_bbox_filter: Whether to filter points using scene bounding box
            max_points: Maximum number of points to triangulate per pair (distributed across pyramid levels)
            use_grid_sampling: Whether to use grid-based sampling
            grid_percentage: Percentage for grid cells
            max_threshold_multiplier: Maximum multiplier for certainty threshold
            pyramid_levels: Number of pyramid levels
            debug: Whether to save debug information
            
        Returns:
            Merged point cloud from all pyramid levels
        """
        img_name1 = self.colmap_dataset.images[img_id1].name
        img_name2 = self.colmap_dataset.images[img_id2].name
        
        # Get image paths
        img_path1 = self.images_dir / img_name1
        img_path2 = self.images_dir / img_name2
        
        logger.info(f"  Multi-resolution processing with {pyramid_levels} pyramid levels")
        
        # Create image pyramids
        pyramid1 = self._create_image_pyramid(str(img_path1), pyramid_levels)
        pyramid2 = self._create_image_pyramid(str(img_path2), pyramid_levels)
        
        # Get original camera information
        info1 = self.colmap_dataset.get_image_info(img_id1)
        info2 = self.colmap_dataset.get_image_info(img_id2)
        K1_orig, K2_orig = info1['K'], info2['K']
        
        # Get original image dimensions
        orig_w1, orig_h1 = get_image_dimensions(str(img_path1))
        orig_w2, orig_h2 = get_image_dimensions(str(img_path2))
        
        point_clouds = []
        
        # Process each pyramid level
        for level in range(pyramid_levels):
            level_img1, level_w1, level_h1 = pyramid1[level]
            level_img2, level_w2, level_h2 = pyramid2[level]
            
            # Calculate points for this level (max_points / 2^level)
            level_max_points = max(10000, max_points // (2 ** level))
            
            logger.info(f"  Processing pyramid level {level}: {level_w1}x{level_h1}, {level_w2}x{level_h2} (max_points: {level_max_points})")
            
            # Scale camera intrinsics for this pyramid level
            K1_scaled = self._scale_intrinsics(K1_orig, (orig_w1, orig_h1), (level_w1, level_h1))
            K2_scaled = self._scale_intrinsics(K2_orig, (orig_w2, orig_h2), (level_w2, level_h2))
            
            if debug:
                logger.info(f"    Intrinsics scaling for level {level}:")
                logger.info(f"      Image 1: {orig_w1}x{orig_h1} -> {level_w1}x{level_h1}")
                logger.info(f"      Original K1: fx={K1_orig[0,0]:.2f}, fy={K1_orig[1,1]:.2f}, cx={K1_orig[0,2]:.2f}, cy={K1_orig[1,2]:.2f}")
                logger.info(f"      Scaled K1:   fx={K1_scaled[0,0]:.2f}, fy={K1_scaled[1,1]:.2f}, cx={K1_scaled[0,2]:.2f}, cy={K1_scaled[1,2]:.2f}")
                logger.info(f"      Image 2: {orig_w2}x{orig_h2} -> {level_w2}x{level_h2}")
                logger.info(f"      Original K2: fx={K2_orig[0,0]:.2f}, fy={K2_orig[1,1]:.2f}, cx={K2_orig[0,2]:.2f}, cy={K2_orig[1,2]:.2f}")
                logger.info(f"      Scaled K2:   fx={K2_scaled[0,0]:.2f}, fy={K2_scaled[1,1]:.2f}, cx={K2_scaled[0,2]:.2f}, cy={K2_scaled[1,2]:.2f}")
            
            # Create temporary image info with scaled intrinsics
            level_info1 = info1.copy()
            level_info2 = info2.copy()
            level_info1['K'] = K1_scaled
            level_info2['K'] = K2_scaled
            
            try:
                # Convert pyramid images to PIL format for matching
                level_img1_pil = self._numpy_to_pil(level_img1)
                level_img2_pil = self._numpy_to_pil(level_img2)
                
                # Perform dense matching at this resolution
                warp, certainty, match_info = self._dense_match_images(
                    level_img1_pil, level_img2_pil, level_w1, level_h1
                )
                
                # Generate point cloud for this level with scaled intrinsics
                level_pcd, _ = self._generate_point_cloud_with_intrinsics(
                    warp, certainty, level_info1, level_info2, img_id1, img_id2,
                    level_img1_pil, level_img2_pil, level_w1, level_h1, level_w2, level_h2,
                    use_bbox_filter=use_bbox_filter, max_points=level_max_points,
                    use_grid_sampling=use_grid_sampling, grid_percentage=grid_percentage,
                    max_threshold_multiplier=max_threshold_multiplier, debug=debug
                )
                
                if len(level_pcd.points) > 0:
                    point_clouds.append(level_pcd)
                    logger.info(f"    Level {level}: Generated {len(level_pcd.points)} points")
                    
                    # Save debug information if requested
                    if debug:
                        self._save_pyramid_debug_info(img_id1, img_id2, level, pair_idx, level_pcd, 
                                                    level_img1_pil, level_img2_pil, 
                                                    warp, certainty)
                else:
                    logger.warning(f"    Level {level}: No points generated")
                
            except Exception as e:
                logger.error(f"    Level {level} failed: {str(e)}")
                continue
        
        # Merge point clouds from all levels
        if point_clouds:
            # Add debug information about point cloud coordinates
            if debug:
                logger.info(f"  Debug: Point cloud coordinate ranges before merging:")
                for level, pcd in enumerate(point_clouds):
                    if len(pcd.points) > 0:
                        points = np.asarray(pcd.points)
                        min_coords = points.min(axis=0)
                        max_coords = points.max(axis=0)
                        mean_coords = points.mean(axis=0)
                        logger.info(f"    Level {level}: {len(points)} points, "
                                  f"X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}] (mean: {mean_coords[0]:.3f}), "
                                  f"Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}] (mean: {mean_coords[1]:.3f}), "
                                  f"Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}] (mean: {mean_coords[2]:.3f})")
            
            merged_pcd = self._merge_point_clouds(point_clouds)
            logger.info(f"  Multi-resolution result: {len(merged_pcd.points)} total points from {len(point_clouds)} levels")
            
            # Debug info for merged result
            if debug and len(merged_pcd.points) > 0:
                points = np.asarray(merged_pcd.points)
                min_coords = points.min(axis=0)
                max_coords = points.max(axis=0)
                mean_coords = points.mean(axis=0)
                logger.info(f"  Debug: Merged point cloud range: "
                          f"X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}] (mean: {mean_coords[0]:.3f}), "
                          f"Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}] (mean: {mean_coords[1]:.3f}), "
                          f"Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}] (mean: {mean_coords[2]:.3f})")
            
            return merged_pcd
        else:
            logger.warning(f"  No point clouds generated from any pyramid level")
            return o3d.geometry.PointCloud()
    
    def _numpy_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if len(img_array.shape) == 3:
            # CHW to HWC
            img_array = img_array.transpose(1, 2, 0)
        elif len(img_array.shape) == 2:
            # Grayscale
            pass
        else:
            # Single channel case
            img_array = img_array.squeeze()
        
        # Ensure uint8 format
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _pil_to_tensor(self, pil_image: Image.Image, target_w: int, target_h: int) -> torch.Tensor:
        """
        Convert PIL image to tensor suitable for RoMa model.
        
        Args:
            pil_image: PIL Image to convert
            target_w: Target width
            target_h: Target height
            
        Returns:
            Tensor in CHW format with values in [0, 1]
        """
        # Resize if needed
        if pil_image.size != (target_w, target_h):
            pil_image = pil_image.resize((target_w, target_h), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(pil_image, dtype=np.float32)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        
        # Convert from HWC to CHW and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1)) / 255.0)
        
        return img_tensor
    
    def _dense_match_images(self, img1_pil: Image.Image, img2_pil: Image.Image, 
                           target_w: int, target_h: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform dense matching on two PIL images at specified resolution.
        """
        # Convert PIL images to tensors using our own method
        img1_tensor = self._pil_to_tensor(img1_pil, target_w, target_h)
        img2_tensor = self._pil_to_tensor(img2_pil, target_w, target_h)
        
        # Stack images for batch processing
        imgs = torch.stack([img1_tensor, img2_tensor], dim=0).to(device)
        
        # Perform matching
        with torch.no_grad():
            warp, certainty = self.roma_model.match(imgs[0][None], imgs[1][None])
            
        # Squeeze batch dimension
        warp = warp.squeeze(0).cpu()
        certainty = certainty.squeeze(0).cpu()
        
        # Calculate statistics
        mean_certainty = torch.mean(certainty).item()
        high_conf_ratio = torch.sum(certainty > 0.3).float() / certainty.numel()
        
        info = {
            'mean_certainty': mean_certainty,
            'high_conf_ratio': high_conf_ratio.item()
        }
        
        return warp, certainty, info
    
    def _generate_point_cloud_with_intrinsics(self, warp: torch.Tensor, certainty: torch.Tensor,
                                            info1: dict, info2: dict, img_id1: int, img_id2: int,
                                            img1_pil: Image.Image, img2_pil: Image.Image,
                                            W1: int, H1: int, W2: int, H2: int,
                                            use_bbox_filter: bool = True, max_points: int = 100000,
                                            use_grid_sampling: bool = False, grid_percentage: float = 10.0,
                                            max_threshold_multiplier: float = 5.0, debug: bool = False) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Generate point cloud using custom intrinsics (for scaled pyramid levels).
        """
        logger.info(f"Generating point cloud for pyramid level {img_id1}-{img_id2}")
        
        # Get camera information from provided info
        K1, K2 = info1['K'], info2['K']
        R_rel, t_rel = self.colmap_dataset.get_relative_pose(img_id1, img_id2)
        
        # Debug: Log initial parameters
        if debug:
            logger.info(f"    Pyramid level triangulation parameters:")
            logger.info(f"      Image sizes: {W1}x{H1}, {W2}x{H2}")
            logger.info(f"      Certainty threshold: {0.3}")
            logger.info(f"      Min triangulation angle: {self.min_triangulation_angle}°")
        
        # Convert PIL images to numpy for triangulation
        img1_np = np.array(img1_pil)
        img2_np = np.array(img2_pil)
        if len(img1_np.shape) == 3:
            img1_np = img1_np.transpose(2, 0, 1)  # HWC to CHW
        else:
            img1_np = img1_np[None, :]  # Add channel dimension
        if len(img2_np.shape) == 3:
            img2_np = img2_np.transpose(2, 0, 1)  # HWC to CHW  
        else:
            img2_np = img2_np[None, :]  # Add channel dimension
        
        # Sample high-certainty correspondences
        certainty_threshold = 0.3  # Fixed threshold for pyramid levels
        mask = certainty > certainty_threshold
        valid_indices = torch.nonzero(mask, as_tuple=False)
        
        H, W = certainty.shape
        if len(valid_indices) > max_points:
            if use_grid_sampling:
                # Grid-based sampling for more even distribution
                valid_indices = self._grid_sample_points(
                    valid_indices, certainty, certainty_threshold, 
                    max_points, grid_percentage, max_threshold_multiplier, H, W
                )
            else:
                # Randomly sample points (original behavior)
                perm = torch.randperm(len(valid_indices))[:max_points]
                valid_indices = valid_indices[perm]
        
        logger.info(f"  Triangulating {len(valid_indices)} points")
        
        # Extract correspondences
        y_coords, x_coords = valid_indices[:, 0], valid_indices[:, 1]
        
        # Points in image 1 (pixel coordinates)
        kpts1 = torch.stack([x_coords, y_coords], dim=1).float()
        
        # Corresponding points in image 2 from warp
        warp_coords = warp[y_coords, x_coords, 2:4]  # Get the second image coordinates
        
        # Convert from normalized coordinates to pixel coordinates
        kpts2_x = W * (warp_coords[:, 0] + 1) / 2
        kpts2_y = H * (warp_coords[:, 1] + 1) / 2
        kpts2 = torch.stack([kpts2_x, kpts2_y], dim=1)
        
        # Convert to numpy
        kpts1_np = kpts1.numpy()
        kpts2_np = kpts2.numpy()
        
        logger.info(f"  Key points: Image 1: {kpts1_np.shape}, Image 2: {kpts2_np.shape}")
        
        # Triangulate points
        with self.profiler.profile("triangulation"):
            points_3d_cam1 = triangulate_points(
                kpts1_np, kpts2_np, K1, K2, R_rel, t_rel
            )
        
        # Filter points
        with self.profiler.profile("point_filtering"):
            # Convert to world coordinates for filtering (triangulated points are in camera 1 coordinates)
            # Get relative pose and transform to world coordinates
            cam_center1 = self.colmap_dataset.images[img_id1].projection_center()
            cam_center2 = self.colmap_dataset.images[img_id2].projection_center()
            
            # Transform points from camera 1 coordinates to world coordinates
            R1, t1 = self.colmap_dataset.get_pose(img_id1)
            
            # COLMAP uses cam_from_world: X_cam = R @ X_world + t
            # So to transform from camera to world: X_world = R^T @ (X_cam - t)
            t1_flat = t1.flatten()  # Ensure t1 is 1D for broadcasting
            points_3d_world = (R1.T @ (points_3d_cam1 - t1_flat).T).T
            
            # Apply triangulation angle filtering first (more important than bbox)
            with self.profiler.profile("triangulation_angle_filtering"):
                if len(points_3d_world) > 0:
                    # Get camera centers for triangulation angle calculation
                    rays1 = points_3d_world - cam_center1
                    rays2 = points_3d_world - cam_center2
                    
                    # Normalize rays
                    rays1_norm = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
                    rays2_norm = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)
                    
                    # Compute triangulation angles
                    cos_angles = np.sum(rays1_norm * rays2_norm, axis=1)
                    cos_angles = np.clip(cos_angles, -1.0, 1.0)
                    
                    # Convert to degrees (triangulation angle = 180° - angle between rays)
                    angles_rad = np.arccos(np.abs(cos_angles))
                    angles_deg = np.degrees(angles_rad)
                    
                    # Filter points with sufficient triangulation angle
                    valid_angle_mask = angles_deg >= self.min_triangulation_angle
                    
                    points_3d_world = points_3d_world[valid_angle_mask]
                    kpts1_np = kpts1_np[valid_angle_mask]
                    
                    logger.info(f"  After triangulation angle filtering: {len(points_3d_world)} points (min angle: {self.min_triangulation_angle}°)")
            
            # Apply depth filtering (remove points too close or too far)
            with self.profiler.profile("depth_filtering"):
                if len(points_3d_world) > 0:
                    depths = np.linalg.norm(points_3d_world, axis=1)
                    valid_depth_mask = (depths > 0.1) & (depths < 100)
                    
                    points_3d_world = points_3d_world[valid_depth_mask]
                    kpts1_np = kpts1_np[valid_depth_mask]
                    
                    logger.info(f"  After depth filtering: {len(points_3d_world)} points")
            
            # Apply pair-specific bounding box filtering
            if use_bbox_filter and len(points_3d_world) > 0:
                try:
                    pair_bbox = self.colmap_dataset.compute_pair_bounding_box(
                        img_id1, img_id2, 
                        min_track_size=self.pair_bbox_min_track_size, 
                        margin_factor=self.pair_bbox_margin
                    )
                    
                    # Filter points using pair-specific bounding box
                    bbox_min, bbox_max = pair_bbox['min'], pair_bbox['max']
                    bbox_filter = (
                        (points_3d_world[:, 0] >= bbox_min[0]) & (points_3d_world[:, 0] <= bbox_max[0]) &
                        (points_3d_world[:, 1] >= bbox_min[1]) & (points_3d_world[:, 1] <= bbox_max[1]) &
                        (points_3d_world[:, 2] >= bbox_min[2]) & (points_3d_world[:, 2] <= bbox_max[2])
                    )
                    
                    points_3d_world = points_3d_world[bbox_filter]
                    kpts1_np = kpts1_np[bbox_filter]
                    
                    logger.info(f"  After pair bbox filtering ({pair_bbox['num_points']} ref points): {len(points_3d_world)} points")
                    
                except ValueError as e:
                    logger.warning(f"  Pair bbox filtering failed: {e}, skipping bbox filter")
                except Exception as e:
                    logger.warning(f"  Pair bbox filtering error: {e}, skipping bbox filter")
            
            # Convert back to camera coordinates for color extraction if needed
            # Transform: cam1_point = R1 @ world_point + t1
            if len(points_3d_world) > 0:
                points_3d_cam1 = (R1 @ points_3d_world.T + t1_flat.reshape(-1, 1)).T
            else:
                points_3d_cam1 = points_3d_world  # Empty array
        
        # Extract colors for points
        with self.profiler.profile("color_extraction"):
            if len(points_3d_world) > 0 and len(kpts1_np) > 0:
                if len(img1_np.shape) == 3 and img1_np.shape[0] >= 3:
                    # RGB image - ensure indices are valid
                    y_indices = np.clip(kpts1_np[:, 1].astype(int), 0, img1_np.shape[1] - 1)
                    x_indices = np.clip(kpts1_np[:, 0].astype(int), 0, img1_np.shape[2] - 1)
                    colors = img1_np[:3, y_indices, x_indices].T / 255.0
                else:
                    # Grayscale - use gray color
                    colors = np.full((len(points_3d_world), 3), 0.5)
            else:
                colors = np.empty((0, 3))
        
        # Create point cloud using world coordinates
        pcd = o3d.geometry.PointCloud()
        if len(points_3d_world) > 0:
            pcd.points = o3d.utility.Vector3dVector(points_3d_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Generate depth map (optional, return None for pyramid levels)
        depth_map = None
        
        return pcd, depth_map
    
    def _save_pyramid_debug_info(self, img_id1: int, img_id2: int, level: int, pair_idx: int,
                                pcd: o3d.geometry.PointCloud, img1_pil: Image.Image, img2_pil: Image.Image,
                                warp: torch.Tensor, certainty: torch.Tensor):
        """
        Save debug information for pyramid levels.
        """
        # Create structured debug directory: debug/pair_XX/resY/
        debug_dir = self.output_dir / "debug" / f"pair_{pair_idx:02d}" / f"res{level}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point cloud
        pcd_path = debug_dir / f"cloud_{img_id1}_{img_id2}.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        
        # Save pyramid level images
        img1_path = debug_dir / f"img1_{img_id1}.jpg"
        img2_path = debug_dir / f"img2_{img_id2}.jpg"
        img1_pil.save(img1_path)
        img2_pil.save(img2_path)
        
        # Save certainty map
        cert_path = debug_dir / f"certainty_{img_id1}_{img_id2}.jpg"
        certainty_vis = torch.stack([certainty, certainty, certainty], dim=0)
        tensor_to_pil(certainty_vis, unnormalize=False).save(cert_path)
        
        # Save resolution info
        info_path = debug_dir / "resolution_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Resolution Level: {level}\n")
            f.write(f"Image 1 size: {img1_pil.size}\n")
            f.write(f"Image 2 size: {img2_pil.size}\n")
            f.write(f"Point cloud points: {len(pcd.points)}\n")
            f.write(f"Certainty shape: {certainty.shape}\n")
            f.write(f"Warp shape: {warp.shape}\n")
            f.write(f"Mean certainty: {torch.mean(certainty).item():.4f}\n")
        
        logger.info(f"  Saved debug info for level {level}: {debug_dir.relative_to(self.output_dir)}")
    
    def process_pair(self, img_id1: int, img_id2: int, pair_idx: int, use_bbox_filter: bool = True, max_points: int = 100000,
                     use_grid_sampling: bool = False, grid_percentage: float = 10.0, max_threshold_multiplier: float = 5.0,
                     multi_res: bool = False, pyramid_levels: int = 3, debug: bool = False) -> o3d.geometry.PointCloud:
        """
        Process a single image pair and return point cloud.
        
        Args:
            img_id1, img_id2: Image IDs to process
            pair_idx: Index of this pair for logging
            use_bbox_filter: Whether to filter points using scene bounding box
            max_points: Maximum number of points to triangulate
            use_grid_sampling: Whether to use grid-based sampling instead of random sampling
            grid_percentage: Percentage of image to use for each grid cell (e.g., 10.0 means 10% => 10x10 grid)
            max_threshold_multiplier: Maximum multiplier for certainty_threshold when grid cells have insufficient points
            multi_res: Whether to use multi-resolution processing
            pyramid_levels: Number of pyramid levels for multi-resolution
            debug: Whether to save debug information
            
        Returns:
            Point cloud generated from the image pair
        """
        
        # Route to multi-resolution processing if enabled
        if multi_res:
            return self._process_pair_multires(
                img_id1, img_id2, pair_idx, use_bbox_filter, max_points,
                use_grid_sampling, grid_percentage, max_threshold_multiplier,
                pyramid_levels, debug
            )
        
        # Dense matching
        warp, certainty, match_info = self.dense_match_pair(img_id1, img_id2)
        
        # Save visualization (if enabled)
        if self.save_visualizations:
            self.save_visualization(warp, certainty, img_id1, img_id2)
        else:
            logger.info(f"  Visualizations not enabled (use --enable_visualizations to save)")
        
        # Generate point cloud
        pcd, depth_map = self.generate_point_cloud(warp, certainty, img_id1, img_id2, 
                                                  use_bbox_filter=use_bbox_filter, max_points=max_points,
                                                  use_grid_sampling=use_grid_sampling, grid_percentage=grid_percentage,
                                                  max_threshold_multiplier=max_threshold_multiplier)
        
        # Save individual point cloud
        pcd_path = self.output_dir / "point_clouds" / f"cloud_{img_id1}_{img_id2}.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        
        # Save depth map if generated
        if depth_map is not None:
            self.save_depth_map(depth_map, img_id1)
        
        # Store metadata for summary
        point_count = len(pcd.points)
        self.point_cloud_metadata.append((img_id1, img_id2, point_count))
        
        logger.info(f"  Saved point cloud: {pcd_path.name} ({point_count} points)")
        
        return pcd
    
    def filter_by_triangulation_angle(self, 
                                     points_3d: np.ndarray, 
                                     colors: np.ndarray,
                                     cam_center1: np.ndarray, 
                                     cam_center2: np.ndarray,
                                     min_angle_degrees: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out 3D points with small triangulation angles for better accuracy.
        
        Args:
            points_3d: (N, 3) array of 3D points in world coordinates
            colors: (N, 3) array of RGB colors  
            cam_center1: (3,) camera center 1 in world coordinates
            cam_center2: (3,) camera center 2 in world coordinates
            min_angle_degrees: minimum triangulation angle in degrees
            
        Returns:
            Filtered points and colors
        """
        if len(points_3d) == 0:
            return points_3d, colors
            
        # Compute rays from camera centers to each 3D point
        rays1 = points_3d - cam_center1  # (N, 3)
        rays2 = points_3d - cam_center2  # (N, 3)
        
        # Normalize rays
        rays1_norm = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2_norm = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)
        
        # Compute triangulation angles using dot product
        # cos(angle) = dot(ray1, ray2) / (|ray1| * |ray2|)
        cos_angles = np.sum(rays1_norm * rays2_norm, axis=1)
        
        # Clamp to valid range for arccos (numerical stability)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        
        # Convert to angles in degrees
        angles_rad = np.arccos(cos_angles)
        angles_deg = np.degrees(angles_rad)
        
        # Filter points with sufficient triangulation angle
        valid_angle_mask = angles_deg >= min_angle_degrees
        
        filtered_points = points_3d[valid_angle_mask]
        filtered_colors = colors[valid_angle_mask]
        
        logger.info(f"  Triangulation angle filter: {len(filtered_points)}/{len(points_3d)} points retained "
                   f"(min_angle: {min_angle_degrees}°, mean_angle: {np.mean(angles_deg[valid_angle_mask]):.1f}°)")
        
        return filtered_points, filtered_colors

    def merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Merge multiple point clouds into one."""
        logger.info(f"\nMerging {len(point_clouds)} point clouds...")
        
        if not point_clouds:
            return o3d.geometry.PointCloud()
        
        # Combine all points and colors
        all_points = []
        all_colors = []
        
        for pcd in point_clouds:
            if len(pcd.points) > 0:
                all_points.append(np.asarray(pcd.points))
                all_colors.append(np.asarray(pcd.colors))
        
        if not all_points:
            return o3d.geometry.PointCloud()
        
        # Create merged point cloud
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        logger.info(f"  Combined point cloud: {len(merged_pcd.points)} points")
        
        # Remove duplicates and outliers
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        
        logger.info(f"  After cleanup: {len(merged_pcd.points)} points")
        
        return merged_pcd
    
    def run(self, 
            min_common_points: int = 100,
            min_baseline: float = 0.1,
            max_baseline: float = 2.0,
            max_pairs_per_image: int = 5,
            use_bbox_filter: bool = True,
            max_points: int = 100000,
            pairs_file: str = "pairs.json",
            use_grid_sampling: bool = False,
            grid_percentage: float = 10.0,
            max_threshold_multiplier: float = 5.0,
            multi_res: bool = False,
            pyramid_levels: int = 3,
            debug: bool = False,
            single_pair: int = None):
        """
        Run the complete dense matching pipeline.
        
        Args:
            min_common_points: Minimum number of shared 3D points for pair selection
            min_baseline: Minimum baseline distance for pair selection
            max_baseline: Maximum baseline distance for pair selection
            max_pairs_per_image: Maximum pairs per image (duplicates are automatically removed)
            use_bbox_filter: Whether to filter points using scene bounding box
            max_points: Maximum number of points to triangulate per pair
            pairs_file: Filename to save pairs information in output directory (default: pairs.json)
            use_grid_sampling: Whether to use grid-based sampling instead of random sampling
            grid_percentage: Percentage of image to use for each grid cell (e.g., 10.0 means 10% => 10x10 grid)
            max_threshold_multiplier: Maximum multiplier for certainty_threshold when grid cells have insufficient points
            multi_res: Whether to use multi-resolution processing with image pyramids
            pyramid_levels: Number of pyramid levels for multi-resolution processing
            debug: Whether to save debug information for pyramid levels
            single_pair: Index of specific pair to process (0-based). If None, processes all pairs.
        """
        logger.info("=== Starting Dense Matching Pipeline ===")
        
        # Select image pairs (and save to JSON in output directory)
        # Always save under output directory, extract filename only for safety
        filename = Path(pairs_file).name  # Extract just the filename part
        save_pairs_path = str(self.output_dir / filename)
        logger.info(f"Pairs will be saved to: {save_pairs_path}")
        
        with self.profiler.profile("pair_selection"):
            pairs = self.select_image_pairs(
                min_common_points=min_common_points,
                min_baseline=min_baseline,
                max_baseline=max_baseline,
                max_pairs_per_image=max_pairs_per_image,
                save_pairs_file=save_pairs_path
            )
        
        if not pairs:
            logger.error("No suitable image pairs found!")
            return
        
        # Show available pairs for debugging
        logger.info(f"Found {len(pairs)} suitable image pairs:")
        for idx, (id1, id2, meta) in enumerate(pairs[:10]):  # Show first 10 pairs
            img1_name = self.colmap_dataset.images[id1].name
            img2_name = self.colmap_dataset.images[id2].name
            logger.info(f"  [{idx}] {img1_name} <-> {img2_name} "
                       f"(baseline: {meta['baseline']:.3f}, "
                       f"common_points: {meta['common_points']})")
        if len(pairs) > 10:
            logger.info(f"  ... and {len(pairs) - 10} more pairs (use --single_pair INDEX to process a specific pair)")
        
        # Process each pair
        point_clouds = []
        total_start_time = time.time()
        total_pairs = len(pairs)
        
        # Limit to single pair for debugging if requested
        if single_pair is not None:
            if single_pair < 0 or single_pair >= len(pairs):
                logger.error(f"Invalid pair index {single_pair}. Available pairs: 0-{len(pairs)-1}")
                return
            
            selected_pair = pairs[single_pair]
            pairs = [selected_pair]
            total_pairs = 1
            img_name1 = self.colmap_dataset.images[selected_pair[0]].name
            img_name2 = self.colmap_dataset.images[selected_pair[1]].name
            logger.info(f"=== DEBUG MODE: Processing only pair {single_pair}: {img_name1} <-> {img_name2} ===")
        
        for i, (img_id1, img_id2, metadata) in enumerate(pairs):
            try:
                img_name1 = self.colmap_dataset.images[img_id1].name
                img_name2 = self.colmap_dataset.images[img_id2].name
                
                # Use the correct pair index for debug purposes
                actual_pair_idx = single_pair if single_pair is not None else i
                
                logger.info(f"\n=== Processing pair {i+1}/{total_pairs}: {img_id1} ({img_name1}) <-> {img_id2} ({img_name2}) ===")
                
                with self.profiler.profile("process_single_pair"):
                    pcd = self.process_pair(img_id1, img_id2, actual_pair_idx, use_bbox_filter=use_bbox_filter, max_points=max_points,
                                          use_grid_sampling=use_grid_sampling, grid_percentage=grid_percentage,
                                          max_threshold_multiplier=max_threshold_multiplier,
                                          multi_res=multi_res, pyramid_levels=pyramid_levels, debug=debug)
                
                if len(pcd.points) > 0:
                    point_clouds.append(pcd)
                else:
                    logger.warning(f"  Empty point cloud generated for pair {i+1}")
                
                # Clear cache every 20 pairs to prevent excessive memory usage in long runs
                if (i + 1) % 20 == 0:
                    cache_stats = self.image_cache.get_stats()
                    if cache_stats['memory_usage_mb'] > 1000:  # If cache uses > 1GB
                        logger.info(f"  Clearing image cache (was using {cache_stats['memory_usage_mb']:.1f} MB)")
                        self.image_cache.clear()
            except Exception as e:
                logger.error(f"  Failed to process pair {i+1}: {e}")
                continue
        
        # Merge point clouds
        if point_clouds:
            with self.profiler.profile("merge_point_clouds"):
                merged_pcd = self.merge_point_clouds(point_clouds)
            
            # Save merged point cloud
            merged_path = self.output_dir / "merged_point_cloud.ply"
            o3d.io.write_point_cloud(str(merged_path), merged_pcd)
            
            total_time = time.time() - total_start_time
            logger.info(f"\n=== Pipeline Complete ===")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Processed {len(point_clouds)} pairs successfully")
            logger.info(f"Final merged point cloud: {len(merged_pcd.points)} points")
            logger.info(f"Saved to: {merged_path}")
            
            # Print profiling summary
            self.profiler.print_summary()
            
            # Print cache statistics
            cache_stats = self.image_cache.get_stats()
            logger.info(f"\n" + "="*60)
            logger.info("IMAGE CACHE STATISTICS")
            logger.info("="*60)
            logger.info(f"Cached image variants: {cache_stats['cached_items']}")
            logger.info(f"Max cache size: {cache_stats['max_size']}")
            logger.info(f"Estimated memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
            logger.info("="*60)
            
            # Save detailed profiling data to JSON
            profiling_path = self.output_dir / "profiling_data.json"
            self.profiler.save_to_file(str(profiling_path))
            logger.info(f"Saved detailed profiling data to: {profiling_path}")
            
            # Save processing summary
            summary_path = self.output_dir / "processing_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Dense Matching Results\n")
                f.write(f"=====================\n\n")
                
                # Write command line arguments
                if self.command_args:
                    f.write(f"Command Line Used:\n")
                    f.write(f"------------------\n")
                    # Reconstruct command line
                    cmd_parts = [sys.argv[0]]
                    if self.command_args.get('scene_dir'):
                        cmd_parts.extend(['-s', str(self.command_args['scene_dir'])])
                    if self.command_args.get('colmap_path') and self.command_args['colmap_path'] != str(Path(self.command_args['scene_dir']) / 'sparse'):
                        cmd_parts.extend(['-c', str(self.command_args['colmap_path'])])
                    if self.command_args.get('images_dir') and self.command_args['images_dir'] != str(Path(self.command_args['scene_dir']) / 'images'):
                        cmd_parts.extend(['-i', str(self.command_args['images_dir'])])
                    if self.command_args.get('output_dir') and self.command_args['output_dir'] != str(Path(self.command_args['scene_dir']) / 'output'):
                        cmd_parts.extend(['-o', str(self.command_args['output_dir'])])
                    if self.command_args.get('model_type') != 'tiny_roma':
                        cmd_parts.extend(['-m', str(self.command_args['model_type'])])
                    if self.command_args.get('resolution') != [864, 1152]:
                        cmd_parts.extend(['-r'] + [str(x) for x in self.command_args['resolution']])
                    if self.command_args.get('enable_visualizations'):
                        cmd_parts.append('-v')


                    # Add other non-default parameters
                    for param in ['min_common_points', 'min_baseline', 'max_baseline', 'max_pairs_per_image', 'max_points', 'min_triangulation_angle']:
                        if param in self.command_args:
                            value = self.command_args[param]
                            defaults = {'min_common_points': 100, 'min_baseline': 0.1, 'max_baseline': 2.0, 
                                      'max_pairs_per_image': 5, 'max_points': 100000, 'min_triangulation_angle': 2.0}
                            if value != defaults.get(param):
                                flag_map = {'min_common_points': '-p', 'min_baseline': '-b', 'max_baseline': '-B',
                                          'max_pairs_per_image': '-n', 'max_points': '-P', 'min_triangulation_angle': '-a'}
                                cmd_parts.extend([flag_map[param], str(value)])
                    
                    f.write(f"  {' '.join(cmd_parts)}\n\n")
                    
                    f.write(f"All Arguments:\n")
                    f.write(f"--------------\n")
                    for key, value in self.command_args.items():
                        f.write(f"  {key}: {value}\n")
                    f.write(f"\n")
                
                f.write(f"Processed Paths:\n")
                f.write(f"----------------\n")
                f.write(f"Input COLMAP path: {self.colmap_path}\n")
                f.write(f"Input images dir: {self.images_dir}\n")
                f.write(f"Output directory: {self.output_dir}\n")
                f.write(f"Model type: {self.model_type}\n")
                f.write(f"Input resolution parameter: {self.resolution}\n")
                if self.actual_resolution:
                    f.write(f"Actual model resolution: {self.actual_resolution} (H, W)\n")
                else:
                    f.write(f"Actual model resolution: Not determined yet\n")
                f.write(f"\n")
                f.write(f"Processed {len(point_clouds)} image pairs\n")
                f.write(f"Final point cloud: {len(merged_pcd.points)} points\n")
                f.write(f"Total processing time: {total_time:.2f}s\n\n")
                
                # Write performance profiling data
                f.write(f"Performance Profiling:\n")
                f.write(f"----------------------\n")
                profiling_summary = self.profiler.get_summary()
                if profiling_summary:
                    # Sort by total time (descending)
                    sorted_ops = sorted(profiling_summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
                    for operation, stats in sorted_ops:
                        f.write(f"  {operation}:\n")
                        f.write(f"    Total time: {stats['total_time']:.3f}s\n")
                        f.write(f"    Average time: {stats['average_time']:.3f}s\n")
                        f.write(f"    Count: {stats['count']}\n")
                        if stats['count'] > 1:
                            f.write(f"    Min/Max: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s\n")
                        f.write(f"\n")
                else:
                    f.write(f"  No profiling data available\n")
                f.write(f"\n")
                
                # Write image cache statistics
                cache_stats = self.image_cache.get_stats()
                f.write(f"Image Cache Statistics:\n")
                f.write(f"-----------------------\n")
                f.write(f"  Cached image variants: {cache_stats['cached_items']}\n")
                f.write(f"  Max cache size: {cache_stats['max_size']}\n")
                f.write(f"  Estimated memory usage: {cache_stats['memory_usage_mb']:.1f} MB\n")
                f.write(f"\n")
                
                f.write(f"Image pairs processed:\n")
                for i, (img_id1, img_id2, point_count) in enumerate(self.point_cloud_metadata):
                    name1 = self.colmap_dataset.images[img_id1].name
                    name2 = self.colmap_dataset.images[img_id2].name
                    f.write(f"  {i+1}. {name1} <-> {name2} {point_count}\n")
        else:
            logger.error("No point clouds generated!")


def main():
    parser = ArgumentParser(description="Dense matching with RoMa and COLMAP integration",
                          formatter_class=ArgumentDefaultsHelpFormatter)
    
    # JSON configuration support
    parser.add_argument("-j", "--json_config", 
                       help="Path to JSON configuration file. CLI arguments override JSON values.")
    parser.add_argument("--create_example_config", 
                       help="Create an example configuration file at the specified path and exit")
    
    parser.add_argument("-s", "--scene_dir", 
                       help="Main scene directory containing images/ and sparse/ subdirectories (required unless using JSON config)")
    parser.add_argument("-c", "--colmap_path", 
                       help="Path to COLMAP sparse reconstruction directory (default: auto-detect scene_dir/sparse/0 or scene_dir/sparse)")
    parser.add_argument("-i", "--images_dir",
                       help="Directory containing input images (default: scene_dir/images)")
    parser.add_argument("-o", "--output_dir",
                       help="Output directory for results (default: scene_dir/output, or relative to scene_dir if specified)")
    parser.add_argument("-m", "--model_type", default=None,
                       choices=["roma_outdoor", "roma_indoor", "tiny_roma"],
                       help="RoMa model type to use (default: tiny_roma)")
    parser.add_argument("-r", "--resolution", nargs=2, type=int, default=None,
                       help="Target resolution for matching [height width] (default: [864, 1152])")
    parser.add_argument("-p", "--min_common_points", type=int, default=None,
                       help="Minimum common 3D points for pair selection (default: 100)")
    parser.add_argument("-b", "--min_baseline", type=float, default=None,
                       help="Minimum baseline distance for pair selection (default: 0.1)")
    parser.add_argument("-B", "--max_baseline", type=float, default=None,
                       help="Maximum baseline distance for pair selection (default: 2.0)")
    parser.add_argument("-n", "--max_pairs_per_image", type=int, default=None,
                       help="Maximum pairs per image (duplicates automatically removed) (default: 5)")
    parser.add_argument("-P", "--max_points", type=int, default=None,
                       help="Maximum number of points to triangulate per pair (default: 100000)")
    parser.add_argument("-f", "--pairs_file", type=str, default=None,
                       help="Filename to save pairs information as JSON in output directory (default: pairs.json)")
    parser.add_argument("-d", "--disable_bbox_filter", action="store_true",
                       help="Disable pair-specific bounding box filtering of point clouds")
    parser.add_argument("--pair_bbox_min_track_size", type=int, default=None,
                       help="Minimum track size for points used in pair bounding box computation (default: 3)")
    parser.add_argument("--pair_bbox_margin", type=float, default=None,
                       help="Margin factor for pair bounding box as fraction of box size (default: 0.1 = 10%%)")
    parser.add_argument("--include_cameras_in_bbox", action="store_true",
                       help="Include camera centers when computing scene bounding box (default: False)")
    parser.add_argument("-a", "--min_triangulation_angle", type=float, default=None,
                       help="Minimum triangulation angle in degrees for point filtering (default: 2.0)")
    parser.add_argument("-v", "--enable_visualizations", action="store_true",
                       help="Enable saving of match visualizations")

    parser.add_argument("--cache_size", type=int, default=None,
                       help="Maximum number of cached image variants (higher = faster but more memory) (default: 100)")
    
    # Grid sampling options
    parser.add_argument("--use_grid_sampling", action="store_true",
                       help="Use grid-based sampling instead of random sampling for more even point distribution")
    parser.add_argument("--grid_percentage", type=float, default=None,
                       help="Percentage of image dimension for each grid cell (e.g., 10.0 means 10%% => 10x10 grid) (default: 10.0)")
    parser.add_argument("--max_threshold_multiplier", type=float, default=None,
                       help="Maximum multiplier for certainty threshold when grid cells have insufficient points (default: 5.0)")
    
    # Multi-resolution options
    parser.add_argument("--multi_res", action="store_true",
                       help="Enable multi-resolution processing using image pyramids")
    parser.add_argument("--pyramid_levels", type=int, default=None,
                       help="Number of pyramid levels for multi-resolution processing (default: 3)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode to save individual resolution point clouds and images")
    parser.add_argument("--single_pair", type=int, default=None,
                       help="Process only the specified pair index (0-based) for debugging purposes. If not set, processes all pairs.")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_example_config:
        create_example_config(args.create_example_config)
        logger.info("Example configuration created. Edit the file and run with -j <config_file>")
        return
    
    # Load and merge configurations
    config = {}
    
    if args.json_config:
        # Load JSON configuration
        json_config = load_json_config(args.json_config)
        
        # Convert args to dict (only include non-None values for CLI override)
        cli_args = {}
        for key, value in vars(args).items():
            if key not in ['json_config', 'create_example_config'] and value is not None:
                cli_args[key] = value
        
        # Check if scene_dir was overridden via CLI
        scene_dir_overridden = 'scene_dir' in cli_args
        
        # Merge configurations (CLI overrides JSON)
        config = merge_configs(json_config, cli_args)
        
        # Store whether scene_dir was overridden for path derivation
        config['_scene_dir_overridden'] = scene_dir_overridden
    else:
        # Use command line arguments only
        config = {key: value for key, value in vars(args).items() 
                 if key not in ['json_config', 'create_example_config']}
        config['_scene_dir_overridden'] = False
    
    # Apply default values for any missing required parameters
    defaults = {
        'model_type': 'tiny_roma',
        'resolution': [864, 1152],
        'min_common_points': 100,
        'min_baseline': 0.1,
        'max_baseline': 2.0,
        'max_pairs_per_image': 5,
        'max_points': 100000,
        'pairs_file': 'pairs.json',
        'disable_bbox_filter': False,
        'min_triangulation_angle': 2.0,
        'enable_visualizations': False,

        'cache_size': 100,
        'use_grid_sampling': False,
        'grid_percentage': 10.0,
        'max_threshold_multiplier': 5.0,
        'multi_res': False,
        'pyramid_levels': 3,
        'debug': False,
        'single_pair': None,
        'include_cameras_in_bbox': False,
        'pair_bbox_min_track_size': 3,
        'pair_bbox_margin': 0.1
    }
    
    for key, default_value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = default_value
    
    # Validate that scene_dir is provided (either via JSON or CLI)
    if not config.get('scene_dir'):
        logger.error("scene_dir is required. Provide it via command line (-s) or JSON configuration file.")
        parser.print_help()
        return
    
    # Derive paths from scene_dir
    scene_dir = Path(config['scene_dir'])
    scene_dir_overridden = config.pop('_scene_dir_overridden', False)
    
    # If scene_dir was overridden via CLI, re-derive all relative paths
    # Otherwise, only set paths that are not already provided
    should_derive_colmap = not config.get('colmap_path') or scene_dir_overridden
    should_derive_images = not config.get('images_dir') or scene_dir_overridden
    should_derive_output = not config.get('output_dir') or scene_dir_overridden
    
    if should_derive_colmap:
        # Try common COLMAP sparse directory patterns
        sparse_candidates = [
            scene_dir / "sparse" / "0",  # Most common: sparse/0
            scene_dir / "sparse",        # Direct sparse folder
            scene_dir / "colmap" / "sparse" / "0",  # Alternative structure
            scene_dir / "colmap" / "sparse",
            scene_dir / "glomap" / "sparse" / "0"
        ]
        
        # Use the first existing candidate
        sparse_found = False
        for candidate in sparse_candidates:
            if candidate.exists() and candidate.is_dir():
                config['colmap_path'] = str(candidate)
                sparse_found = True
                break
        
        if not sparse_found:
            config['colmap_path'] = str(scene_dir / "sparse")  # Default fallback
            
        if scene_dir_overridden:
            logger.info(f"Re-derived colmap_path due to scene_dir override: {config['colmap_path']}")
    
    if should_derive_images:
        config['images_dir'] = str(scene_dir / "images")
        if scene_dir_overridden:
            logger.info(f"Re-derived images_dir due to scene_dir override: {config['images_dir']}")
    
    if should_derive_output:
        config['output_dir'] = str(scene_dir / "output")
        if scene_dir_overridden:
            logger.info(f"Re-derived output_dir due to scene_dir override: {config['output_dir']}")
    else:
        # If output_dir is provided and not being re-derived, make it relative to scene_dir if needed
        if not Path(config['output_dir']).is_absolute():
            config['output_dir'] = str(scene_dir / config['output_dir'])
    
    # Now convert to ConfigObj after all path resolution is complete
    class ConfigObj:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = ConfigObj(config)
    
    # Log the derived paths
    logger.info(f"Scene directory: {scene_dir}")
    logger.info(f"COLMAP path: {args.colmap_path}")
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate inputs
    if not scene_dir.exists():
        logger.error(f"Scene directory does not exist: {scene_dir}")
        return
    
    if not Path(args.colmap_path).exists():
        logger.error(f"COLMAP path does not exist: {args.colmap_path}")
        return
    
    if not Path(args.images_dir).exists():
        logger.error(f"Images directory does not exist: {args.images_dir}")
        return
    
    # Use the merged configuration
    command_args = config.copy()
    
    # Save the final configuration to output directory
    save_config_to_output(command_args, args.output_dir)
    
    # Create pipeline
    pipeline = DenseMatchingPipeline(
        colmap_path=args.colmap_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        resolution=tuple(args.resolution),
        min_triangulation_angle=args.min_triangulation_angle,
        save_visualizations=args.enable_visualizations,

        command_args=command_args,
        cache_size=args.cache_size,
        include_cameras_in_bbox=config['include_cameras_in_bbox'],
        pair_bbox_min_track_size=config['pair_bbox_min_track_size'],
        pair_bbox_margin=config['pair_bbox_margin']
    )
    
    # Run pipeline
    pipeline.run(
        min_common_points=config['min_common_points'],
        min_baseline=config['min_baseline'],
        max_baseline=config['max_baseline'],
        max_pairs_per_image=config['max_pairs_per_image'],
        use_bbox_filter=not config['disable_bbox_filter'],
        max_points=config['max_points'],
        pairs_file=config['pairs_file'],
        use_grid_sampling=config['use_grid_sampling'],
        grid_percentage=config['grid_percentage'],
        max_threshold_multiplier=config['max_threshold_multiplier'],
        multi_res=config['multi_res'],
        pyramid_levels=config['pyramid_levels'],
        debug=config['debug'],
        single_pair=config['single_pair']
    )


if __name__ == "__main__":
    main() 