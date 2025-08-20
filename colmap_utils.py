"""
COLMAP utilities for reading calibration data and selecting image pairs.
Uses pycolmap package for efficient COLMAP database access.
"""

import numpy as np
import pycolmap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import cv2

logger = logging.getLogger(__name__)

class COLMAPDataset:
    """
    Handler for COLMAP reconstruction data.
    
    Includes caching of visible 3D points per image for faster point queries.
    """
    
    def __init__(self, colmap_path: str, build_cache: bool = True):
        """
        Initialize COLMAP dataset.
        
        Args:
            colmap_path: Path to COLMAP reconstruction directory containing:
                - cameras.bin/cameras.txt
                - images.bin/images.txt  
                - points3D.bin/points3D.txt
            build_cache: Whether to build visible points cache during initialization (default: True)
        """
        self.colmap_path = Path(colmap_path)
        self.reconstruction = None
        self.images = {}
        self.cameras = {}
        self.points3d = {}
        # Cache for visible 3D points per image (for performance)
        self._visible_points_cache = {}
        self._cache_built = False
        
        self._load_reconstruction()
        # Build cache during initialization for better performance
        if build_cache:
            self._build_visible_points_cache()
    
    def _load_reconstruction(self):
        """Load COLMAP reconstruction from binary or text files."""
        try:
            # Try binary format first
            self.reconstruction = pycolmap.Reconstruction(str(self.colmap_path))
            logger.info(f"Loaded COLMAP reconstruction from {self.colmap_path}")
        except Exception as e:
            logger.error(f"Failed to load COLMAP reconstruction: {e}")
            raise
        
        # Extract data for easier access
        # pycolmap reconstruction objects are already dictionaries keyed by ID
        self.images = dict(self.reconstruction.images)
        self.cameras = dict(self.reconstruction.cameras) 
        self.points3d = dict(self.reconstruction.points3D)
        
        logger.info(f"Loaded {len(self.images)} images, {len(self.cameras)} cameras, {len(self.points3d)} 3D points")
    
    def _build_visible_points_cache(self):
        """Build cache of visible 3D point IDs for each image."""
        logger.info("Building visible points cache for faster point queries...")
        
        self._visible_points_cache = {}
        
        for img_id, image in self.images.items():
            visible_points = set()
            for point2d in image.points2D:
                if point2d.has_point3D():
                    visible_points.add(point2d.point3D_id)
            self._visible_points_cache[img_id] = visible_points
        
        self._cache_built = True
        logger.info(f"Built visible points cache for {len(self.images)} images")
    
    def _get_visible_points(self, img_id: int) -> set:
        """Get cached visible 3D point IDs for an image."""
        if not self._cache_built:
            self._build_visible_points_cache()
        return self._visible_points_cache.get(img_id, set())
    
    def clear_cache(self):
        """Clear the visible points cache to free memory."""
        self._visible_points_cache.clear()
        self._cache_built = False
        logger.info("Cleared visible points cache")
    
    def rebuild_cache(self):
        """Rebuild the visible points cache (useful if data has changed)."""
        self.clear_cache()
        self._build_visible_points_cache()
    
    def get_image_by_name(self, name: str) -> Optional[pycolmap.Image]:
        """Get image by filename."""
        for img in self.images.values():
            if img.name == name:
                return img
        return None
    
    def get_camera_matrix(self, camera_id: int) -> np.ndarray:
        """Get camera intrinsic matrix K."""
        camera = self.cameras[camera_id]
        
        # Get model name from enum
        model_name = str(camera.model).split('.')[-1]  # Extract name from enum
        
        if model_name in ['SIMPLE_PINHOLE', 'PINHOLE']:
            if model_name == 'SIMPLE_PINHOLE':
                f, cx, cy = camera.params[:3]
                fx = fy = f
            else:  # PINHOLE
                fx, fy, cx, cy = camera.params[:4]
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        elif model_name in ['SIMPLE_RADIAL', 'RADIAL']:
            f, cx, cy = camera.params[:3]
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Unsupported camera model: {model_name}")
        
        return K
    
    def get_pose(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera pose as rotation matrix and translation vector.
        
        Returns:
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
        """
        image = self.images[image_id]
        
        # Get pose from cam_from_world transformation
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation.reshape(3, 1)
        
        return R, t
    
    def get_relative_pose(self, img_id1: int, img_id2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get relative pose from image 1 to image 2.
        
        Returns:
            R_rel: 3x3 relative rotation matrix
            t_rel: 3x1 relative translation vector
        """
        R1, t1 = self.get_pose(img_id1)
        R2, t2 = self.get_pose(img_id2)
        
        # Compute relative pose: T_rel = T2 * T1^-1
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        return R_rel, t_rel
    
    def get_common_points(self, img_id1: int, img_id2: int) -> List[int]:
        """Get 3D point IDs visible in both images."""
        # Use cached visible points for much faster lookup
        points1 = self._get_visible_points(img_id1)
        points2 = self._get_visible_points(img_id2)
        
        return list(points1.intersection(points2))
    
    def get_num_visible_points(self, img_id: int) -> int:
        """Get number of visible 3D points in an image."""
        return len(self._get_visible_points(img_id))
    
    def get_visible_points_list(self, img_id: int) -> List[int]:
        """Get list of visible 3D point IDs in an image."""
        return list(self._get_visible_points(img_id))
    
    def compute_baseline(self, img_id1: int, img_id2: int) -> float:
        """Compute baseline distance between two cameras."""
        _, t1 = self.get_pose(img_id1)
        _, t2 = self.get_pose(img_id2)
        
        return np.linalg.norm(t2 - t1)
    
    def select_image_pairs(self, 
                          min_common_points: int = 100,
                          min_baseline: float = 0.1,
                          max_baseline: float = 2.0,
                          max_pairs_per_image: int = 5,
                          save_pairs_file: Optional[str] = None) -> List[Tuple[int, int, Dict]]:
        """
        Select good image pairs for dense matching using per-image selection:
        - For each image, select up to max_pairs_per_image best quality pairs
        - Remove duplicate pairs from the combined selection
        - Process all unique pairs found
        
        Args:
            min_common_points: Minimum number of shared 3D points
            min_baseline: Minimum baseline distance
            max_baseline: Maximum baseline distance  
            max_pairs_per_image: Maximum pairs per image to select
            save_pairs_file: Optional filename to save pairs information as JSON
            
        Returns:
            List of (img_id1, img_id2, metadata) tuples sorted by ascending image IDs
        """
        image_ids = list(self.images.keys())
        logger.info(f"Selecting image pairs for {len(image_ids)} images (max {max_pairs_per_image} pairs per image)")
        
        # Helper function to create pair metadata
        def create_pair_metadata(img_id1: int, img_id2: int) -> Dict:
            common_points = self.get_common_points(img_id1, img_id2)
            baseline = self.compute_baseline(img_id1, img_id2)
            R_rel, _ = self.get_relative_pose(img_id1, img_id2)
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            angle_deg = np.degrees(angle)
            quality = len(common_points) * baseline / (1 + angle_deg / 30)
            
            return {
                'common_points': len(common_points),
                'baseline': baseline,
                'angle_deg': angle_deg,
                'quality': quality,
                'point_ids': common_points
            }
        
        # Use a set to automatically handle duplicate removal
        # Store as (min_id, max_id, metadata) to ensure consistent ordering
        unique_pairs = {}
        
        # For each image, find its best pairs
        for img_id in image_ids:
            candidates_for_image = []
            
            # Find all valid pairs involving this image
            for other_img_id in image_ids:
                if img_id == other_img_id:
                    continue
                
                # Check common points
                common_points = self.get_common_points(img_id, other_img_id)
                if len(common_points) < min_common_points:
                    continue
                
                # Check baseline
                baseline = self.compute_baseline(img_id, other_img_id)
                if baseline < min_baseline or baseline > max_baseline:
                    continue
                
                # Create metadata and add to candidates
                metadata = create_pair_metadata(img_id, other_img_id)
                candidates_for_image.append((img_id, other_img_id, metadata))
            
            # Sort by quality and take the best ones
            candidates_for_image.sort(key=lambda x: x[2]['quality'], reverse=True)
            best_pairs = candidates_for_image[:max_pairs_per_image]
            
            # Add to unique pairs set (with consistent ordering to avoid duplicates)
            for img_id1, img_id2, metadata in best_pairs:
                pair_key = (min(img_id1, img_id2), max(img_id1, img_id2))
                if pair_key not in unique_pairs or unique_pairs[pair_key][2]['quality'] < metadata['quality']:
                    unique_pairs[pair_key] = (pair_key[0], pair_key[1], metadata)
        
        # Convert to list and sort by ascending image IDs (primary), then by quality (secondary)
        selected_pairs = list(unique_pairs.values())
        selected_pairs.sort(key=lambda x: (x[0], x[1], -x[2]['quality']))
        
        # Log statistics
        image_pair_counts = {img_id: 0 for img_id in image_ids}
        for img_id1, img_id2, _ in selected_pairs:
            image_pair_counts[img_id1] += 1
            image_pair_counts[img_id2] += 1
        
        covered_images = len([img_id for img_id in image_ids if image_pair_counts[img_id] > 0])
        avg_pairs_per_image = np.mean([count for count in image_pair_counts.values() if count > 0]) if covered_images > 0 else 0
        
        logger.info(f"Selected {len(selected_pairs)} unique image pairs from {len(image_ids)} images")
        logger.info(f"Coverage: {covered_images}/{len(image_ids)} images have at least one pair")
        logger.info(f"Average pairs per covered image: {avg_pairs_per_image:.1f}")
        
        # Log pair distribution
        pair_counts = list(image_pair_counts.values())
        logger.info(f"Pair distribution - Min: {min(pair_counts)}, Max: {max(pair_counts)}, "
                   f"Images with 0 pairs: {pair_counts.count(0)}")
        
        logger.info(f"Pairs sorted by ascending image IDs")
        
        # Save pairs information to JSON file if requested
        if save_pairs_file:
            pairs_data = []
            for img_id1, img_id2, metadata in selected_pairs:
                pair_info = {
                    'image_id1': int(img_id1),
                    'image_id2': int(img_id2), 
                    'image_name1': self.images[img_id1].name,
                    'image_name2': self.images[img_id2].name,
                    'common_points': metadata['common_points'],
                    'baseline': float(metadata['baseline']),
                    'angle_deg': float(metadata['angle_deg']),
                    'quality': float(metadata['quality'])
                }
                pairs_data.append(pair_info)
            
            save_path = Path(save_pairs_file)
            with open(save_path, 'w') as f:
                json.dump(pairs_data, f, indent=2)
            logger.info(f"Saved {len(selected_pairs)} pairs information to {save_path}")
        
        return selected_pairs
    
    def get_image_path(self, image_id: int, images_dir: str) -> Path:
        """Get full path to image file."""
        image = self.images[image_id]
        return Path(images_dir) / image.name
    
    def get_image_info(self, image_id: int) -> Dict:
        """Get comprehensive image information."""
        image = self.images[image_id]
        camera = self.cameras[image.camera_id]
        R, t = self.get_pose(image_id)
        K = self.get_camera_matrix(image.camera_id)
        
        return {
            'id': image_id,
            'name': image.name,
            'camera_id': image.camera_id,
            'width': camera.width,
            'height': camera.height,
            'camera_model': str(camera.model).split('.')[-1],
            'camera_params': camera.params,
            'K': K,
            'R': R,
            't': t,
            'num_points3d': image.num_points3D
        }
    
    def get_camera_centers(self) -> np.ndarray:
        """
        Get all camera center positions in world coordinates.
        
        Returns:
            camera_centers: (N, 3) array of camera center positions
        """
        camera_centers = []
        
        for image in self.images.values():
            # Camera center in world coordinates using projection_center method
            center = image.projection_center()
            camera_centers.append(center)
        
        return np.array(camera_centers)
    
    def get_filtered_3d_points(self, min_visibility: int = 3) -> Tuple[np.ndarray, List[int]]:
        """
        Get 3D points that are visible in at least min_visibility images.
        
        Args:
            min_visibility: Minimum number of images that must see the point
            
        Returns:
            points: (N, 3) array of filtered 3D point coordinates
            point_ids: List of corresponding point IDs
        """
        filtered_points = []
        filtered_ids = []
        
        for point_id, point in self.points3d.items():
            # Count how many images see this point (from track elements)
            visibility_count = len(point.track.elements)
            
            if visibility_count >= min_visibility:
                filtered_points.append(point.xyz)
                filtered_ids.append(point_id)
        
        if filtered_points:
            return np.array(filtered_points), filtered_ids
        else:
            return np.empty((0, 3)), []
    
    def compute_scene_bounding_box(self, 
                                  min_visibility: int = 3,
                                  include_cameras: bool = True,
                                  margin_factor: float = 0.1,
                                  robust_percentile: float = 95.0) -> Dict:
        """
        Robustly compute scene bounding box using 3D points and camera positions.
        
        Args:
            min_visibility: Minimum visibility for 3D points to be included
            include_cameras: Whether to include camera positions in bounding box
            margin_factor: Additional margin as fraction of box size (0.1 = 10% margin)
            robust_percentile: Percentile for robust outlier removal (95.0 = remove 5% outliers)
            
        Returns:
            Dictionary containing bounding box information:
            - 'min': (3,) minimum coordinates [x, y, z]
            - 'max': (3,) maximum coordinates [x, y, z] 
            - 'center': (3,) center coordinates [x, y, z]
            - 'size': (3,) box dimensions [width, height, depth]
            - 'volume': scalar volume of the bounding box
            - 'num_points': number of 3D points used
            - 'num_cameras': number of cameras used
            - 'margin_applied': margin factor used
        """
        logger.info(f"Computing scene bounding box (min_visibility={min_visibility})")
        
        # Collect all points for bounding box computation
        all_points = []
        
        # Get filtered 3D points
        points_3d, point_ids = self.get_filtered_3d_points(min_visibility)
        if len(points_3d) > 0:
            all_points.append(points_3d)
            logger.info(f"  Using {len(points_3d)} 3D points (min visibility: {min_visibility})")
        else:
            logger.warning(f"  No 3D points found with minimum visibility {min_visibility}")
        
        # Get camera centers
        if include_cameras:
            camera_centers = self.get_camera_centers()
            if len(camera_centers) > 0:
                all_points.append(camera_centers)
                logger.info(f"  Using {len(camera_centers)} camera positions")
        
        if not all_points:
            logger.error("No points available for bounding box computation")
            raise ValueError("No valid points found for bounding box computation")
        
        # Combine all points
        combined_points = np.vstack(all_points)
        logger.info(f"  Total points for bounding box: {len(combined_points)}")
        
        # Robust outlier removal using percentiles
        if robust_percentile < 100.0 and len(combined_points) > 10:
            # Calculate percentiles for each axis
            lower_percentile = (100.0 - robust_percentile) / 2
            upper_percentile = 100.0 - lower_percentile
            
            lower_bounds = np.percentile(combined_points, lower_percentile, axis=0)
            upper_bounds = np.percentile(combined_points, upper_percentile, axis=0)
            
            # Filter points within percentile bounds
            mask = np.all((combined_points >= lower_bounds) & 
                         (combined_points <= upper_bounds), axis=1)
            filtered_points = combined_points[mask]
            
            logger.info(f"  Robust filtering: {len(filtered_points)}/{len(combined_points)} points retained "
                       f"({robust_percentile}th percentile)")
            combined_points = filtered_points
        
        # Compute basic bounding box
        min_coords = np.min(combined_points, axis=0)
        max_coords = np.max(combined_points, axis=0)
        
        # Apply margin
        if margin_factor > 0:
            box_size = max_coords - min_coords
            margin = box_size * margin_factor
            min_coords -= margin
            max_coords += margin
            logger.info(f"  Applied {margin_factor:.1%} margin to bounding box")
        
        # Compute derived properties
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        volume = np.prod(size)
        
        bbox_info = {
            'min': min_coords,
            'max': max_coords, 
            'center': center,
            'size': size,
            'volume': volume,
            'num_points': len(points_3d) if len(points_3d) > 0 else 0,
            'num_cameras': len(self.images) if include_cameras else 0,
            'margin_applied': margin_factor,
            'robust_percentile': robust_percentile
        }
        
        logger.info(f"  Bounding box computed:")
        logger.info(f"    Min: [{min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}]")
        logger.info(f"    Max: [{max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f}]")
        logger.info(f"    Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
        logger.info(f"    Volume: {volume:.6f}")
        
        return bbox_info
    
    def is_point_in_bbox(self, point: np.ndarray, bbox: Dict) -> bool:
        """
        Check if a 3D point is inside the bounding box.
        
        Args:
            point: (3,) coordinates to test
            bbox: Bounding box dictionary from compute_scene_bounding_box()
            
        Returns:
            True if point is inside bounding box
        """
        return np.all((point >= bbox['min']) & (point <= bbox['max']))
    
    def filter_points_by_bbox(self, points: np.ndarray, bbox: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter 3D points to only include those inside bounding box.
        
        Args:
            points: (N, 3) array of 3D points
            bbox: Bounding box dictionary
            
        Returns:
            filtered_points: (M, 3) points inside bbox
            mask: (N,) boolean mask indicating which points are inside
        """
        mask = np.all((points >= bbox['min']) & (points <= bbox['max']), axis=1)
        return points[mask], mask
    
    def get_bbox_corners(self, bbox: Dict) -> np.ndarray:
        """
        Get the 8 corner points of the bounding box.
        
        Args:
            bbox: Bounding box dictionary
            
        Returns:
            corners: (8, 3) array of corner coordinates
        """
        min_coords = bbox['min']
        max_coords = bbox['max']
        
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # min corner
            [max_coords[0], min_coords[1], min_coords[2]],  
            [min_coords[0], max_coords[1], min_coords[2]], 
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]]   # max corner
        ])
        
        return corners
    
    def export_bbox_to_ply(self, bbox: Dict, output_path: str):
        """
        Export bounding box as wireframe PLY file for visualization.
        
        Args:
            bbox: Bounding box dictionary
            output_path: Path to save PLY file
        """
        corners = self.get_bbox_corners(bbox)
        
        # Define edges of the bounding box (wireframe)
        edges = [
            # Bottom face
            [0, 1], [1, 3], [3, 2], [2, 0],
            # Top face  
            [4, 5], [5, 7], [7, 6], [6, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Write PLY file
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(corners)}\n")
            f.write("property float x\n")
            f.write("property float y\n") 
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            # Write vertices (red color for bounding box)
            for corner in corners:
                f.write(f"{corner[0]:.6f} {corner[1]:.6f} {corner[2]:.6f} 255 0 0\n")
            
            # Write edges
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")
        
        logger.info(f"Exported bounding box to {output_path}")


def triangulate_points( kpts1: np.ndarray, 
                        kpts2: np.ndarray,
                        K1: np.ndarray,
                        K2: np.ndarray, 
                        R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
    """
    Fast triangulation using OpenCV (if available).
    
    Args:
        kpts1: (N, 2) keypoints in image 1
        kpts2: (N, 2) keypoints in image 2  
        K1, K2: (3, 3) camera intrinsic matrices
        R: (3, 3) relative rotation matrix (cam1 to cam2)
        t: (3, 1) relative translation vector (cam1 to cam2)
        
    Returns:
        points3d: (N, 3) triangulated 3D points in camera 1 coordinate system
    """
    if len(kpts1) == 0:
        return np.empty((0, 3))
    
    # Camera projection matrices
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t])
    
    # OpenCV expects points as (2, N) arrays
    points1 = kpts1.T  # (2, N)
    points2 = kpts2.T  # (2, N)
    
    # Triangulate using OpenCV (much faster than manual DLT)
    points4d = cv2.triangulatePoints(P1, P2, points1, points2)
    
    # Convert from homogeneous coordinates to 3D
    # Handle potential division by zero
    w = points4d[3, :]
    valid_mask = np.abs(w) > 1e-8
    
    points3d = np.zeros((len(kpts1), 3))
    points3d[valid_mask, :] = (points4d[:3, valid_mask] / w[valid_mask]).T
    
    # For invalid points, set to origin
    points3d[~valid_mask] = 0.0
    
    return points3d
