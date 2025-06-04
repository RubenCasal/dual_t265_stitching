# 📷 DualFisheyeStitcher Class - Real-Time Fisheye Image Stitching 

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

### `estimate_overlap_ssim_partial`

This method estimates the horizontal overlap percentage between the left and right dewarped images by focusing exclusively on their bordering regions. Specifically, it compares the right edge of the left image and the left edge of the right image using SSIM to find the best match.

It iteratively shifts the overlapping edge patches over a specified max_offset range and calculates the SSIM score for each alignment. The shift yielding the highest SSIM score determines the best horizontal alignment. The overlap ratio is then computed as the number of overlapping pixels (best_offset) divided by the image width.

**Returns**: A float representing the overlap percentage in the range [0, 1].

---
### `estimate_vertical_misalignment`

This method estimates vertical misalignment (in pixels) between two grayscale images by sliding one image vertically relative to the other and computing SSIM for each offset.

It searches within a max_shift range in both directions (up/down) and returns the shift (dy) that maximizes similarity. A positive return value indicates the right image should be shifted downward for alignment.

**Returns**: An integer dy representing the number of pixels to shift the right image vertically.

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

### `build_partial_equirectangular_map`

This method generates an undistortion map to convert a fisheye image into a partial equirectangular projection (ERP), useful for panoramic or wide-angle image processing pipelines.

Unlike a full ERP that covers 360° horizontally and 180° vertically, this method maps only a selected field of view (`fov_deg`) into a 2D plane, producing a compact and computationally efficient representation. The generated remap matrices are intended to be used with `cv2.remap()`.

**Parameters:**

* `K_fisheye (np.ndarray)`: The 3x3 intrinsic matrix of the fisheye camera.
* `D_fisheye (np.ndarray)`: The 4-element distortion coefficients (k1, k2, k3, k4).
* `output_size (tuple)`: Desired (width, height) of the undistorted image.
* `fov_deg (tuple)`: Horizontal and vertical field of view in degrees. Default is (120, 90).
* `cx, cy (float, optional)`: Override the camera principal point. If not provided, values from `K_fisheye` are used.

**How it works:**

* Each output pixel is interpreted as a direction vector on a unit sphere, defined by azimuth (`phi`) and elevation (`theta`) angles.
* These spherical directions are converted to 3D Cartesian coordinates.
* The direction vectors are projected into the distorted fisheye plane using the inverse of the fisheye projection model.
* Distortion is applied analytically using the standard polynomial fisheye model.
* Final distorted coordinates are mapped into pixel coordinates using the intrinsic matrix.

**Returns:** A tuple of remap matrices `(map_x, map_y)` for `cv2.remap()` to produce the ERP view.

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
