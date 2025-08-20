#!/usr/bin/env python3
"""
Dense Matching with RoMa and COLMAP Integration

This script performs dense matching between image pairs using RoMa matcher,
with automatic pair selection based on COLMAP calibration data.
Generates point clouds from matching results and merges them.

Usage:
    python demo_dense_matching.py --colmap_path /path/to/colmap/sparse/0 \
                                  --images_dir /path/to/images \
                                  --output_dir /path/to/output
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import open3d as o3d

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from colmap_utils import COLMAPDataset, triangulate_points
from romatch import roma_outdoor, tiny_roma_v1_outdoor
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


class DenseMatchingPipeline:
    """Pipeline for dense matching and point cloud generation."""
    
    def __init__(self, 
                 colmap_path: str,
                 images_dir: str,
                 output_dir: str,
                 model_type: str = "roma_outdoor",
                 resolution: Tuple[int, int] = (864, 1152),
                 min_triangulation_angle: float = 2.0,
                 save_visualizations: bool = True):
        """
        Initialize dense matching pipeline.
        
        Args:
            colmap_path: Path to COLMAP sparse reconstruction
            images_dir: Directory containing images
            output_dir: Output directory for results
            model_type: RoMa model type ("roma_outdoor" or "tiny_roma")
            resolution: Target resolution for matching
        """
        self.colmap_path = Path(colmap_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.min_triangulation_angle = min_triangulation_angle
        self.save_visualizations = save_visualizations
        
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
        elif model_type == "tiny_roma":
            self.roma_model = tiny_roma_v1_outdoor(device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        
        # Storage for point clouds
        self.point_clouds = []
        
        # Scene bounding box (computed once)
        self.scene_bbox = None
        
    def select_image_pairs(self, 
                          min_common_points: int = 100,
                          min_baseline: float = 0.1,
                          max_baseline: float = 2.0,
                          max_pairs: Optional[int] = 20,
                          max_pairs_per_image: int = 5) -> List[Tuple[int, int, Dict]]:
        """Select good image pairs for dense matching with balanced coverage."""
        logger.info("Selecting image pairs...")
        pairs = self.colmap_dataset.select_image_pairs(
            min_common_points=min_common_points,
            min_baseline=min_baseline, 
            max_baseline=max_baseline,
            max_pairs=max_pairs,
            max_pairs_per_image=max_pairs_per_image
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
                                  include_cameras: bool = True,
                                  margin_factor: float = 0.15) -> Dict:
        """Compute scene bounding box from COLMAP data."""
        if self.scene_bbox is None:
            logger.info("Computing scene bounding box for point cloud filtering...")
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
        
        # Load and resize images
        start_time = time.time()
        
        if self.model_type == "roma_outdoor":
            H, W = self.roma_model.get_output_resolution()
            img1 = Image.open(img_path1).resize((W, H))
            img2 = Image.open(img_path2).resize((W, H))
            
            # Perform matching
            warp, certainty = self.roma_model.match(str(img_path1), str(img_path2), device=device)
            
        else:  # tiny_roma
            # For tiny RoMa, we need to resize images manually for memory control
            max_size = 800  # Limit maximum dimension to control memory usage
            
            img1 = Image.open(img_path1)
            img2 = Image.open(img_path2)
            
            # Resize while maintaining aspect ratio
            def resize_with_max_size(img, max_size):
                w, h = img.size
                if max(w, h) > max_size:
                    if w > h:
                        new_w, new_h = max_size, int(h * max_size / w)
                    else:
                        new_w, new_h = int(w * max_size / h), max_size
                    return img.resize((new_w, new_h))
                return img
            
            img1_resized = resize_with_max_size(img1, max_size)
            img2_resized = resize_with_max_size(img2, max_size)
            
            # Save temporarily and match
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1, \
                 tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
                img1_resized.save(tmp1.name)
                img2_resized.save(tmp2.name)
                
                try:
                    warp, certainty = self.roma_model.match(tmp1.name, tmp2.name)
                    H, W = warp.shape[:2]
                    logger.info(f"  Warp shape: {warp.shape}")
                    logger.info(f"  Certainty shape: {certainty.shape}")
                    logger.info(f"  Resized image 1: {img1_resized.size}")
                    logger.info(f"  Resized image 2: {img2_resized.size}")
                    
                    # Clean up temporary files
                    os.unlink(tmp1.name)
                    os.unlink(tmp2.name)
                    
                except Exception as e:
                    logger.error(f"  Error during matching: {e}")
                    # Clean up temporary files even on error
                    try:
                        os.unlink(tmp1.name)
                        os.unlink(tmp2.name)
                    except:
                        pass
                    raise
        
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
    
    def generate_point_cloud(self,
                           warp: torch.Tensor,
                           certainty: torch.Tensor,
                           img_id1: int,
                           img_id2: int,
                           certainty_threshold: float = 0.3,
                           max_points: int = 100000,
                           use_bbox_filter: bool = True) -> o3d.geometry.PointCloud:
        """
        Generate point cloud from dense matching results.
        
        Args:
            warp: Dense correspondence field
            certainty: Matching certainty map
            img_id1, img_id2: Image IDs
            certainty_threshold: Minimum certainty for triangulation
            max_points: Maximum number of points to triangulate
        """
        logger.info(f"Generating point cloud for pair {img_id1}-{img_id2}")
        
        # Get camera information
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
        
        # Load original images for colors
        img_path1 = self.colmap_dataset.get_image_path(img_id1, self.images_dir)
        img_path2 = self.colmap_dataset.get_image_path(img_id2, self.images_dir)
        
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        
        # Get original image sizes
        orig_w1, orig_h1 = img1.size
        orig_w2, orig_h2 = img2.size
        
        # Get warp resolution and ensure images match exactly
        H, W = warp.shape[:2]
        # For tiny roma, ensure we resize to match warp output exactly
        img1_resized = img1.resize((W, H), Image.LANCZOS)
        img2_resized = img2.resize((W, H), Image.LANCZOS)
        
        # Convert to numpy
        img1_np = np.array(img1_resized) / 255.0
        img2_np = np.array(img2_resized) / 255.0
        
        # Sample high-certainty correspondences
        mask = certainty > certainty_threshold
        valid_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(valid_indices) > max_points:
            # Randomly sample points
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
        points_3d = triangulate_points(kpts1_np, kpts2_np, K1_scaled, K2_scaled, R_rel, t_rel)
        
        # Transform points from camera 1 coordinate system to world coordinates
        R1, t1 = self.colmap_dataset.get_pose(img_id1)
        # COLMAP uses cam_from_world: X_cam = R @ X_world + t
        # So to transform from camera to world: X_world = R^T @ (X_cam - t)
        t1_flat = t1.flatten()  # Ensure t1 is 1D for broadcasting
        points_3d_world = (R1.T @ (points_3d - t1_flat).T).T
        
        # Use world coordinates for further processing
        points_3d = points_3d_world
        
        logger.info(f"  Point cloud center: [{np.mean(points_3d, axis=0)[0]:.3f}, {np.mean(points_3d, axis=0)[1]:.3f}, {np.mean(points_3d, axis=0)[2]:.3f}]")
        
        # Get colors from image 1
        colors = img1_np[y_coords.cpu().numpy(), x_coords.cpu().numpy()]
        
        # Filter out points that are too close or too far (using world coordinates)
        depths = np.linalg.norm(points_3d, axis=1)
        valid_depth_mask = (depths > 0.1) & (depths < 100)
        
        points_3d = points_3d[valid_depth_mask]
        colors = colors[valid_depth_mask]
        
        logger.info(f"  After depth filtering: {len(points_3d)} points")
        
        # Filter points by triangulation angle (remove points with small viewing angles)
        points_3d, colors = self.filter_by_triangulation_angle(
            points_3d, colors, cam_center1, cam_center2, min_angle_degrees=self.min_triangulation_angle
        )
        
        # Optional: Filter by scene bounding box
        if use_bbox_filter and len(points_3d) > 0:
            scene_bbox = self.compute_scene_bounding_box()
            if scene_bbox is not None:
                filtered_points, bbox_mask = self.colmap_dataset.filter_points_by_bbox(points_3d, scene_bbox)
                filtered_colors = colors[bbox_mask]
                
                logger.info(f"  Bounding box filter: {len(filtered_points)}/{len(points_3d)} points retained")
                points_3d = filtered_points
                colors = filtered_colors
        
        logger.info(f"  Final point cloud: {len(points_3d)} points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd
    
    def save_visualization(self, 
                          warp: torch.Tensor,
                          certainty: torch.Tensor,
                          img_id1: int,
                          img_id2: int):
        """Save matching visualization."""
        img_path1 = self.colmap_dataset.get_image_path(img_id1, self.images_dir)
        img_path2 = self.colmap_dataset.get_image_path(img_id2, self.images_dir)
        
        H, W = warp.shape[:2]
        
        # Load and resize images
        img1 = Image.open(img_path1).resize((W, H))
        img2 = Image.open(img_path2).resize((W, H))
        
        # Convert to tensors
        x1 = (torch.tensor(np.array(img1)) / 255).to(device).permute(2, 0, 1)
        x2 = (torch.tensor(np.array(img2)) / 255).to(device).permute(2, 0, 1)
        
        # Create warped visualization
        if self.model_type == "roma_outdoor":
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
    
    def process_pair(self, img_id1: int, img_id2: int, pair_idx: int, use_bbox_filter: bool = True) -> o3d.geometry.PointCloud:
        """Process a single image pair and return point cloud."""
        logger.info(f"\n=== Processing pair {pair_idx + 1}: {img_id1} <-> {img_id2} ===")
        
        # Dense matching
        warp, certainty, match_info = self.dense_match_pair(img_id1, img_id2)
        
        # Save visualization (if enabled)
        if self.save_visualizations:
            self.save_visualization(warp, certainty, img_id1, img_id2)
        else:
            logger.info(f"  Visualizations disabled")
        
        # Generate point cloud
        pcd = self.generate_point_cloud(warp, certainty, img_id1, img_id2, use_bbox_filter=use_bbox_filter)
        
        # Save individual point cloud
        pcd_path = self.output_dir / "point_clouds" / f"cloud_{img_id1}_{img_id2}.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        logger.info(f"  Saved point cloud: {pcd_path.name} ({len(pcd.points)} points)")
        
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
            max_pairs: Optional[int] = 20,
            max_pairs_per_image: int = 5,
            use_bbox_filter: bool = True):
        """Run the complete dense matching pipeline."""
        logger.info("=== Starting Dense Matching Pipeline ===")
        
        # Select image pairs
        pairs = self.select_image_pairs(
            min_common_points=min_common_points,
            min_baseline=min_baseline,
            max_baseline=max_baseline,
            max_pairs=max_pairs,
            max_pairs_per_image=max_pairs_per_image
        )
        
        if not pairs:
            logger.error("No suitable image pairs found!")
            return
        
        # Process each pair
        point_clouds = []
        total_start_time = time.time()
        
        for i, (img_id1, img_id2, metadata) in enumerate(pairs):
            try:
                pcd = self.process_pair(img_id1, img_id2, i, use_bbox_filter=use_bbox_filter)
                if len(pcd.points) > 0:
                    point_clouds.append(pcd)
                else:
                    logger.warning(f"  Empty point cloud generated for pair {i+1}")
            except Exception as e:
                logger.error(f"  Failed to process pair {i+1}: {e}")
                continue
        
        # Merge point clouds
        if point_clouds:
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
            
            # Save processing summary
            summary_path = self.output_dir / "processing_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Dense Matching Results\n")
                f.write(f"=====================\n\n")
                f.write(f"Input COLMAP path: {self.colmap_path}\n")
                f.write(f"Input images dir: {self.images_dir}\n")
                f.write(f"Output directory: {self.output_dir}\n")
                f.write(f"Model type: {self.model_type}\n")
                f.write(f"Resolution: {self.resolution}\n\n")
                f.write(f"Processed {len(point_clouds)} image pairs\n")
                f.write(f"Final point cloud: {len(merged_pcd.points)} points\n")
                f.write(f"Total processing time: {total_time:.2f}s\n\n")
                f.write(f"Image pairs processed:\n")
                for i, (id1, id2, meta) in enumerate(pairs[:len(point_clouds)]):
                    name1 = self.colmap_dataset.images[id1].name
                    name2 = self.colmap_dataset.images[id2].name
                    f.write(f"  {i+1}. {name1} <-> {name2}\n")
        else:
            logger.error("No point clouds generated!")


def main():
    parser = ArgumentParser(description="Dense matching with RoMa and COLMAP integration")
    parser.add_argument("--colmap_path", required=True, 
                       help="Path to COLMAP sparse reconstruction directory")
    parser.add_argument("--images_dir", required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--model_type", default="roma_outdoor",
                       choices=["roma_outdoor", "tiny_roma"],
                       help="RoMa model type to use")
    parser.add_argument("--resolution", nargs=2, type=int, default=[864, 1152],
                       help="Target resolution for matching [height width]")
    parser.add_argument("--min_common_points", type=int, default=100,
                       help="Minimum common 3D points for pair selection")
    parser.add_argument("--min_baseline", type=float, default=0.1,
                       help="Minimum baseline distance for pair selection")
    parser.add_argument("--max_baseline", type=float, default=2.0,
                       help="Maximum baseline distance for pair selection")
    parser.add_argument("--max_pairs", type=int, default=20,
                       help="Maximum number of pairs to process")
    parser.add_argument("--max_pairs_per_image", type=int, default=5,
                       help="Maximum number of pairs per image for balanced coverage")
    parser.add_argument("--disable_bbox_filter", action="store_true",
                       help="Disable bounding box filtering of point clouds")
    parser.add_argument("--min_triangulation_angle", type=float, default=2.0,
                       help="Minimum triangulation angle in degrees for point filtering (default: 2.0)")
    parser.add_argument("--disable_visualizations", action="store_true",
                       help="Disable saving of match visualizations to save space and speed")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.colmap_path).exists():
        logger.error(f"COLMAP path does not exist: {args.colmap_path}")
        return
    
    if not Path(args.images_dir).exists():
        logger.error(f"Images directory does not exist: {args.images_dir}")
        return
    
    # Create pipeline
    pipeline = DenseMatchingPipeline(
        colmap_path=args.colmap_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        resolution=tuple(args.resolution),
        min_triangulation_angle=args.min_triangulation_angle,
        save_visualizations=not args.disable_visualizations
    )
    
    # Run pipeline
    pipeline.run(
        min_common_points=args.min_common_points,
        min_baseline=args.min_baseline,
        max_baseline=args.max_baseline,
        max_pairs=args.max_pairs,
        max_pairs_per_image=args.max_pairs_per_image,
        use_bbox_filter=not args.disable_bbox_filter
    )


if __name__ == "__main__":
    main() 