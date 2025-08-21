# Dense Matching with RoMa and COLMAP Integration

This directory contains a comprehensive dense matching pipeline that integrates RoMa (Robust Matching) with COLMAP calibration data to generate high-quality 3D point clouds from image pairs.

## Features

🎯 **Automatic Image Pair Selection** - Intelligently selects optimal image pairs with balanced coverage:
- Each image appears in at least one pair (ensuring full dataset coverage)
- Each image appears in at most N pairs (controlling computational load)
- Prioritizes quality while maintaining coverage balance
- Considers common 3D points, baseline distance, and viewing angles

🔍 **Dense Matching** - Uses RoMa models for robust dense correspondence:
- Support for both `roma_outdoor` and `tiny_roma` models
- GPU acceleration with CUDA 12.8 support
- Adaptive resolution handling

🌐 **3D Point Cloud Generation** - Creates detailed point clouds from matches:
- Triangulation using camera calibration from COLMAP
- **Triangulation angle filtering** - Removes points with poor geometric constraints
- Color mapping from source images
- Statistical outlier removal
- Automatic filtering of invalid depths
- **Smart bounding box filtering** using sparse 3D points and camera positions

🔗 **Point Cloud Merging** - Combines multiple point clouds:
- Voxel-based downsampling to remove duplicates
- Statistical outlier filtering
- Merged point cloud export in PLY format

## Quick Start

### Prerequisites

1. **Environment Setup** - Use the provided setup script:
```bash
./setup_environment.sh
conda activate roma_env
```

2. **COLMAP Reconstruction** - You need a COLMAP sparse reconstruction with:
   - `cameras.bin/cameras.txt`
   - `images.bin/images.txt`
   - `points3D.bin/points3D.txt`

3. **Image Directory** - Original images used in COLMAP reconstruction

### Basic Usage

```bash
python demo/demo_dense_matching.py \
    --colmap_path /path/to/colmap/sparse/0 \
    --images_dir /path/to/images \
    --output_dir /path/to/output
```

### Advanced Usage

```bash
python demo/demo_dense_matching.py \
    --colmap_path /path/to/colmap/sparse/0 \
    --images_dir /path/to/images \
    --output_dir /path/to/output \
    --model_type roma_outdoor \
    --resolution 864 1152 \
    --min_common_points 150 \
    --min_baseline 0.2 \
    --max_baseline 1.5 \
    --max_pairs 15 \
    --max_pairs_per_image 3 \
    --disable_bbox_filter
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--colmap_path` | Path to COLMAP sparse reconstruction directory | Required |
| `--images_dir` | Directory containing input images | Required |
| `--output_dir` | Output directory for results | Required |
| `--model_type` | RoMa model (`roma_outdoor`, `roma_indoor`, or `tiny_roma`) | `roma_outdoor` |
| `--resolution` | Target resolution for matching [height width] | `[864, 1152]` |
| `--min_common_points` | Minimum common 3D points for pair selection | `100` |
| `--min_baseline` | Minimum baseline distance for pair selection | `0.1` |
| `--max_baseline` | Maximum baseline distance for pair selection | `2.0` |
| `--max_pairs` | Maximum number of pairs to process | `20` |
| `--max_pairs_per_image` | Maximum pairs per image for balanced coverage | `5` |
| `--disable_bbox_filter` | Disable bounding box filtering of point clouds | `False` |
| `--min_triangulation_angle` | Minimum triangulation angle in degrees for point filtering | `2.0` |
| `--disable_visualizations` | Disable saving of match visualizations to save space and speed | `False` |

## Output Structure

The script creates the following output structure:

```
output_dir/
├── merged_point_cloud.ply          # Final merged point cloud
├── scene_bounding_box.ply          # Scene bounding box visualization
├── processing_summary.txt          # Processing statistics
├── point_clouds/                   # Individual point clouds
│   ├── cloud_1_5.ply
│   ├── cloud_2_8.ply
│   └── ...
├── visualizations/                 # Matching visualizations
│   ├── match_1_5.jpg              # Warped matching visualization
│   ├── certainty_1_5.jpg          # Certainty map
│   └── ...
└── matches/                        # Raw matching data (future use)
```

## Implementation Details

### COLMAP Integration (`colmap_utils.py`)

The `COLMAPDataset` class provides:
- **Camera Model Support**: PINHOLE, SIMPLE_PINHOLE, RADIAL, SIMPLE_RADIAL
- **Pose Computation**: Relative poses between camera pairs
- **3D Point Analysis**: Common point detection and baseline calculation
- **Quality Scoring**: Automated pair selection based on geometric criteria
- **🆕 Bounding Box Computation**: Robust scene bounds using sparse 3D points with minimum visibility filtering

### Dense Matching Pipeline

1. **Balanced Pair Selection Algorithm**: 
   - **Phase 1 (Coverage)**: Ensures every image appears in at least one pair
     - Iterates through quality-sorted candidates
     - Prioritizes pairs that help cover uncovered images
     - Respects the `max_pairs_per_image` limit
   - **Phase 2 (Quality)**: Fills remaining slots with highest quality pairs
     - Adds additional pairs for images under the limit
     - Maintains quality ranking while respecting constraints
   - **Result**: Balanced coverage where all images participate without overloading any single image

2. **Dense Matching**:
   - Uses RoMa models for robust dense correspondence
   - Handles different resolutions and model types
   - Provides certainty maps for quality assessment

3. **Triangulation**:
   - Uses DLT (Direct Linear Transformation) for 3D point calculation
   - Scales camera matrices for different image resolutions
   - Filters points based on depth validity

4. **Point Cloud Processing**:
   - Colors points using source image data
   - **Smart filtering** using scene bounding box (computed from sparse 3D points with ≥3 visibility)
   - Removes statistical outliers  
   - Performs voxel downsampling for efficiency

### Supported Camera Models

The pipeline supports COLMAP camera models:
- **PINHOLE**: `fx, fy, cx, cy`
- **SIMPLE_PINHOLE**: `f, cx, cy`
- **RADIAL**: `f, cx, cy, k1`
- **SIMPLE_RADIAL**: `f, cx, cy, k1`

## Performance Tips

1. **GPU Memory**: Use smaller resolutions for large image datasets
2. **Quality vs Speed**: Adjust `certainty_threshold` (default: 0.3)
3. **Point Density**: Modify `max_points` parameter (default: 100,000)
4. **Coverage vs Load**: Adjust `max_pairs_per_image` to balance coverage and computation
5. **Point Cloud Quality**: Enable `--disable_bbox_filter` if scene bounds are too restrictive
6. **Pair Selection**: Tune baseline parameters for your scene scale

## Key Algorithms

### Balanced Pair Selection

For a dataset with 10 images and `max_pairs_per_image=3`:

**Traditional approach**: Might select pairs (1,2), (1,3), (1,4), (2,3), (2,4) - leaving images 5-10 without pairs

**Balanced approach**: 
- **Phase 1**: Ensures pairs like (1,5), (2,6), (3,7), (4,8), (9,10) to cover all images
- **Phase 2**: Adds high-quality pairs (1,2), (3,4), etc. up to the limit
- **Result**: All 10 images participate, each in 1-3 pairs

### Smart Bounding Box Filtering

**Problem**: Dense matching can generate outlier points far from the actual scene

**Solution**: Use COLMAP sparse reconstruction to define scene bounds:
1. **Filter sparse 3D points** by visibility (≥3 cameras see each point)
2. **Include camera positions** for complete scene coverage  
3. **Robust outlier removal** using percentile filtering (remove 5% extremes)
4. **Add safety margin** (15% expansion) for edge cases
5. **Filter dense point clouds** to remove unrealistic triangulations

## Example Workflow

1. **Run COLMAP** on your image dataset:
```bash
colmap feature_extractor --database_path database.db --image_path images/
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images/ --output_path sparse/
```

2. **Run Dense Matching**:
```bash
python demo/demo_dense_matching.py \
    --colmap_path sparse/0 \
    --images_dir images/ \
    --output_dir dense_results/
```

3. **View Results**:
```bash
# View point cloud
open3d.visualization.draw_geometries([o3d.io.read_point_cloud("dense_results/merged_point_cloud.ply")])

# View with bounding box for context  
pcd = o3d.io.read_point_cloud("dense_results/merged_point_cloud.ply")
bbox = o3d.io.read_point_cloud("dense_results/scene_bounding_box.ply")
open3d.visualization.draw_geometries([pcd, bbox])

# Or use CloudCompare, MeshLab, etc.
```

## Troubleshooting

**Empty Point Clouds**: 
- Reduce `certainty_threshold` 
- Check camera calibration quality
- Verify image pair has sufficient overlap

**Memory Issues**:
- Reduce `max_points` parameter
- Use smaller resolution
- Process fewer pairs at once

**Poor Quality Results**:
- Increase `min_common_points`
- Adjust baseline thresholds
- Check COLMAP reconstruction quality

**Uneven Coverage**:
- Increase `max_pairs_per_image` for more redundancy
- Decrease `max_pairs_per_image` if some images dominate
- Check if isolated images have sufficient common points

**Missing Point Cloud Data**:
- Check if bounding box is too restrictive (view `scene_bounding_box.ply`)
- Use `--disable_bbox_filter` to see full unfiltered point cloud
- Adjust COLMAP sparse reconstruction quality

## Integration with Other Tools

The generated PLY files are compatible with:
- **Open3D** (Python visualization/processing)
- **CloudCompare** (Professional point cloud software)
- **MeshLab** (3D mesh processing)
- **Blender** (3D modeling and visualization)
- **MATLAB** (via PLY reading functions)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{edstedt2023roma,
  title={Roma: Robust dense feature matching}, 
  author={Edstedt, Johan and Athanasiadis, Ioannis and Wadenb{\"a}ck, M{\aa}rten and Felsberg, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
``` 