# 📷 DualFisheyeStitcher Class - Real-Time Fisheye Image Stitching skere

##  Introduction

The `DualFisheyeStitcher` class is designed to perform real-time dewarping and stitching of fisheye images from a dual-camera setup. It assumes both cameras are positioned on the same horizontal plane and have a horizontal field-of-view overlap.

Although this implementation has been tailored for the Intel RealSense T265, it is fully adaptable to any pair of fisheye cameras as long as their intrinsic and distortion parameters are known.

The class handles lens distortion correction, stereographic or rectilinear dewarping, gain-adjusted blending, and seamless image stitching. All processing steps are optimized for real-time execution, making it ideal for robotics, panoramic vision systems, or immersive computer vision applications.


##  **Clone and build**

```bash
cd ~/ros2_ws/src
git clone https://github.com/RubenCasal/dual_t265_stitching.git
cd ~/ros2_ws
colcon build --packages-select dual_t265_stitching
source install/setup.bash
```

---

## Class Initialization

### **Parameters**

* `frame_width` / `frame_height`: Dimensions of the raw fisheye images.
* `K_cam1`, `K_cam2`: 3×3 intrinsic matrices for camera 1 and camera 2 respectively.
* `D_cam1`, `D_cam2`: 4-element fisheye distortion coefficients (k1, k2, k3, k4).
* `fov_h1`, `fov_v1`, `fov_h2`, `fov_v2`: Horizontal and vertical field of view values for both cameras, in degrees.
* `overlaping_region` *(optional)*: Proportion of the image width shared by both cameras (e.g. `0.3` for 30%).
* `blending_ratio` *(optional)*: Proportion used to define the width of the transition zone for blending (e.g. `0.1`).
* `vertical_correction` *(optional)*: Number of pixels to vertically shift one image to correct misalignment.

### **Behavior**

* Precomputes undistortion LUTs using stereographic projection with soft-FOV reduction to avoid black borders.
* Rotates the LUTs of one camera by 180° to align both views in a common reference frame.
* If the optional stitching parameters are provided, it precomputes:

  * the number of pixels to blend (`blend_px`)
  * the number of pixels to trim (`trim_px`)
  * the cosine-based alpha blending map (`alpha`)
  * and vertical correction offset.

These precomputations enable real-time stitching with high accuracy and performance.
## Calibration Methods

This class provides tools to automatically calibrate the stitching between two fisheye cameras using SSIM-based analysis. These methods allow us to estimate both horizontal and vertical misalignments in the captured views.

### `compute_overlap_ssim`

This method estimates the horizontal overlap percentage between the left and right dewarped images. It works by sliding one image over the other (horizontally) across a defined range (`max_offset`) and computes the Structural Similarity Index (SSIM) for each alignment. The shift that yields the highest SSIM is considered the best overlap position. The overlap ratio is computed as the absolute value of this shift divided by the image width.

---
### `estimate_vertical_shift_ssim`

After determining the horizontal overlap, this method finds the optimal vertical alignment (dy) between the overlapping regions of the two images. It extracts the overlapping band from both images and applies vertical shifts to one of them. For each shift, it computes the SSIM with the other image. The vertical displacement that produces the highest SSIM score is returned as the optimal `dy`. This value ensures that both views are vertically aligned for seamless stitching.

---
### `save_calibration_result`

Once the horizontal and vertical alignment parameters are estimated, this method allows you to persist them into a text file for future use. It appends the calculated overlap ratio and vertical shift (`dy`) into the file specified. Useful for debugging, logging, or applying the calibration in later sessions without recalculating.

---
### LaunchFiles
```bash
ros2 launch dual_t265_stitching dual_fisheye_launch.py
```

### Run Nodes
```bash
ros2 run dual_t265_stitching dual_t265_node
ros2 run dual_t265_stitching overlap_calibration.py
```

## Dewarping Methods

This class implements multiple fisheye dewarping methods using precomputed lookup tables (LUTs) for fast remapping and distortion correction.

---

### `build_stereographic_undistort_map_soft_fov(...)`

Builds a remap (LUT) for stereographic projection, reducing the field of view to suppress edge distortions. It uses the intrinsic and distortion parameters of a fisheye camera and creates a mapping from an ideal stereographic projection to distorted fisheye coordinates. This allows you to later apply `cv2.remap()` efficiently.

Internally:

* Converts each undistorted pixel to spherical coordinates.
* Applies a fisheye distortion model.
* Computes the corresponding distorted coordinates using the fisheye camera intrinsics.
* Returns two matrices `map_x` and `map_y` used for remapping.

---

### `fast_equirectangular_dewarping(frame, camera_id)`

Applies a precomputed remap (from `build_stereographic_undistort_map_soft_fov`) to transform a fisheye image into a stereographic projection.

* Selects the remap based on the camera ID.
* Applies `cv2.remap()` for efficient pixel-wise undistortion.

This method is optimized for real-time performance.

---
## Stitching Methods



### `stitch_blend_optimized(left_img, right_img)`

Stitches two dewarped grayscale fisheye images into a seamless panoramic image using linear alpha blending and automatic gain correction.


**How it works:**

* Crops each image symmetrically around the overlap region.
* Applies a vertical shift (precomputed via SSIM calibration) to one of the images for alignment.
* Performs brightness/gain correction by matching the average intensity in the overlapping band.
* Blends the overlapping region with a cosine-weighted alpha mask for smooth transition.
* Concatenates the non-overlapping parts of the two images into the final stitched result.

This method ensures both geometric alignment and photometric consistency, minimizing visible seams or brightness mismatches.


### LaunchFiles
```bash
ros2 launch dual_t265_stitching dual_fisheye_launch.py
```

### Run Nodes
```bash
ros2 run dual_t265_stitching dual_t265_node
ros2 run dual_t265_stitching stitcher_node.py
```

## Workflow
