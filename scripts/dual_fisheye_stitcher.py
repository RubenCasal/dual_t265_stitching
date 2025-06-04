import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim


class DualFisheyeStitcher:
    def __init__(self, frame_width, frame_height,K_cam1, D_cam1, fov_h1, fov_v1, K_cam2, D_cam2, fov_h2, fov_v2, overlaping_region=None,blending_ratio=None, vertical_correction=None):
    
        # Calibration parameter

        self.overlaping_region = overlaping_region
        self.vertical_correction = vertical_correction
        self.blending_ratio = blending_ratio

        self.output_size = (960,720)
        self.dewarped_width = 960
        self.dewarped_height = 720
        # Shape of the raw fisheye frames
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Calibration Parameters
        # Camera 1
        self.K_cam1 = K_cam1
        self.D_cam1 = D_cam1
        self.fov_h1 = fov_h1
        self.fov_v1 = fov_v1
        self.fov1 = (self.fov_v1, self.fov_h1)

        # Camera 2
        self.K_cam2 = K_cam2
        self.D_cam2 = D_cam2
        self.fov_h2 = fov_h2
        self.fov_v2 = fov_v2
        self.fov2 = (self.fov_v2, self.fov_h2)

        
        # Precompute LUT for camera1
        self.map_x1, self.map_y1 = self.build_partial_equirectangular_map(self.K_cam1, self.D_cam1,self.output_size, self.fov1)

        # Precompute LUT for camera2
        self.map_x2, self.map_y2 = self.build_partial_equirectangular_map(self.K_cam2, self.D_cam2,self.output_size, self.fov2)

        # Rotate 180 degrees one of the cameras
        self.map_x1 = cv2.flip(self.map_x1, -1)
        
        self.map_y1 = cv2.flip(self.map_y1, -1)



        ############### STITCHING PARAMETERS INITIALISATION #############
        if self.overlaping_region and self.blending_ratio and self.vertical_correction:
            # Blending ratio
            self.blending_ratio = blending_ratio

            # Regions calculations
            overlap_px = int(self.frame_width * self.overlaping_region)
            self.blend_px = int(self.frame_width * self.blending_ratio)
            self.trim_px = overlap_px - self.blend_px
            self.trim_half = self.trim_px // 2

            # Blending LUT
            self.alpha_lut = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, self.blend_px))).astype(np.float32)
            alpha = self.alpha_lut[:self.blend_px]
            self.alpha = np.tile(alpha, (self.dewarped_height, 1))
    




    ############   CALIBRATION METHODS    #############
    
    def estimate_vertical_misalignment(left_img: np.ndarray, right_img: np.ndarray, max_shift: int = 50) -> int:
        """
        Estimates vertical misalignment (in pixels) between two grayscale images using SSIM.

        Args:
            left_img (np.ndarray): Left image.
            right_img (np.ndarray): Right image.
            max_shift (int): Maximum vertical shift in pixels to test.

        Returns:
            int: Optimal vertical offset (positive means right image should shift down).
        """
        assert left_img.shape == right_img.shape, "Images must have same shape"
        h, w = left_img.shape

        best_score = -1.0
        best_offset_y = 0

        for dy in range(-max_shift, max_shift + 1):
            if dy < 0:
                l_patch = left_img[:h + dy, :]
                r_patch = right_img[-dy:, :]
            elif dy > 0:
                l_patch = left_img[dy:, :]
                r_patch = right_img[:h - dy, :]
            else:
                l_patch = left_img
                r_patch = right_img

            if l_patch.shape[0] < 10:
                continue

            score = ssim(
                l_patch.astype(np.float32) / 255.0,
                r_patch.astype(np.float32) / 255.0,
                data_range=1.0
            )
            if score > best_score:
                best_score = score
                best_offset_y = dy

        return best_offset_y
    def estimate_overlap_ssim_partial(self, left_img, right_img, max_offset=200):
        """
        Compute horizontal overlap (%) between two grayscale images by comparing
        only the right edge of left_img and the left edge of right_img using SSIM.

        Args:
            left_img (np.ndarray): Left grayscale image.
            right_img (np.ndarray): Right grayscale image.
            max_offset (int): Maximum horizontal shift to test (in pixels).

        Returns:
            overlap_pct: overlap percentage (0–1).
        """
        assert left_img.shape == right_img.shape, "Images must have the same size"
        h, w = left_img.shape

        # Extract only the right edge of left_img and left edge of right_img
        left_edge = left_img[:, w - max_offset : w]
        right_edge = right_img[:, :max_offset]

        best_score = -1.0
        best_offset = 0

        # Test dx from 1 to max_offset
        for dx in range(1, min(max_offset, w) + 1):
            l_patch = left_edge[:, -dx:]
            r_patch = right_edge[:, :dx]

            if l_patch.shape[1] < 10 or r_patch.shape[1] < 10:
                continue

            score = ssim(
                l_patch.astype(np.float32) / 255.0,
                r_patch.astype(np.float32) / 255.0,
                data_range=1.0
            )

            if score > best_score:
                best_score = score
                best_offset = dx

        overlap_px = best_offset
        overlap_pct = overlap_px / w

        return overlap_pct


    


    def save_calibration_result(self, overlap_pct: float, dy: int, file_path: str = "/home/rcasal/ros2_ws/src/dual_t265_stitching/overlap_calibration.txt"):
        """
        Saves horizontal overlap and vertical shift to a text file.

        Args:
            overlap_pct (float): Horizontal overlap ratio.
            dy (int): Vertical shift.
            file_path (str): File path to save the calibration result.
        """
        try:
            with open(file_path, "a") as f:
                f.write(f"Horizontal Overlap: {overlap_pct:.4f}, Vertical Shift (dy): {dy}\n")
            print(f"💾 Calibration result saved to {file_path}")
        except Exception as e:
            print(f"❌ Failed to write calibration result: {e}")


    ####################### DEWARP METHODS #################

    def build_partial_equirectangular_map(
    self,
    K_fisheye,
    D_fisheye,
    output_size,
    fov_deg=(120, 90),  # horizontal, vertical FOV en grados
    cx=None,
    cy=None
):
        """
        Builds an undistortion map using partial equirectangular projection.

        Args:
            K_fisheye (np.ndarray): Intrinsic matrix of the fisheye camera.
            D_fisheye (np.ndarray): Distortion coefficients (k1, k2, k3, k4).
            output_size (tuple): (width, height) of the ERP output image.
            fov_deg (tuple): (horizontal, vertical) FOV in degrees.
            cx, cy (float, optional): Principal point (override K_fisheye).

        Returns:
            tuple[np.ndarray, np.ndarray]: Remap matrices (map_x, map_y).
        """
        width, height = output_size
        fov_x = np.radians(fov_deg[0])
        fov_y = np.radians(fov_deg[1])

        cx = float(K_fisheye[0, 2]) if cx is None else float(cx)
        cy = float(K_fisheye[1, 2]) if cy is None else float(cy)
        fx = float(K_fisheye[0, 0])
        fy = float(K_fisheye[1, 1])
        k1, k2, k3, k4 = D_fisheye.flatten()

        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            theta = (y / (height - 1) - 0.5) * fov_y
            for x in range(width):
                phi = (x / (width - 1) - 0.5) * fov_x

                X = np.cos(theta) * np.sin(phi)
                Y = np.sin(theta)
                Z = np.cos(theta) * np.cos(phi)

                r = np.arccos(Z)
                if r == 0:
                    x_dist = 0
                    y_dist = 0
                else:
                    norm = r / np.sqrt(X**2 + Y**2)
                    x_theta = X * norm
                    y_theta = Y * norm

                    r2 = x_theta**2 + y_theta**2
                    theta_d = r * (1 + k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4)

                    scale = theta_d / r if r != 0 else 1.0
                    x_dist = x_theta * scale
                    y_dist = y_theta * scale

                u = fx * x_dist + cx
                v = fy * y_dist + cy

                map_x[y, x] = u
                map_y[y, x] = v

        return map_x, map_y


    def build_stereographic_undistort_map_soft_fov(self,
            K_fisheye,
            D_fisheye,
            output_size,
            fov_deg=(90, 60),
            fov_margin=0.99,  # 90% of original FOV
            cx=None,
            cy=None
        ):
        """
        Builds stereographic undistortion map with reduced FOV to suppress edge artifacts.

        Args:
            K_fisheye (np.ndarray): Intrinsic matrix of the fisheye camera.
            D_fisheye (np.ndarray): Distortion coefficients (k1, k2, k3, k4).
            output_size (tuple): (width, height) of the undistorted output.
            fov_deg (tuple): Desired (horizontal, vertical) FOV in degrees.
            fov_margin (float): Fraction of original FOV to use (e.g., 0.90 = 90%).
            cx (float, optional): Optical center x (overrides K_fisheye if provided).
            cy (float, optional): Optical center y (overrides K_fisheye if provided).

        Returns:
            tuple[np.ndarray, np.ndarray]: Remap matrices (map_x, map_y) for cv2.remap.
        """
        width, height = output_size
        fov_x, fov_y = np.radians(fov_deg[0] * fov_margin), np.radians(fov_deg[1] * fov_margin)

        fx = (width / 2) / np.tan(fov_x / 2)
        fy = (height / 2) / np.tan(fov_y / 2)
        cx_out = width / 2
        cy_out = height / 2

        cx = float(K_fisheye[0, 2]) if cx is None else float(cx)
        cy = float(K_fisheye[1, 2]) if cy is None else float(cy)

        k1, k2, k3, k4 = [float(k) for k in D_fisheye.flatten()]
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                x_r = (x - cx_out) / fx
                y_r = (y - cy_out) / fy
                r = np.sqrt(x_r ** 2 + y_r ** 2)

                theta = 2 * np.arctan(r / 2) if r != 0 else 0.0
                theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

                scale = theta_d / r if r != 0 else 0.0
                x_f = x_r * scale
                y_f = y_r * scale

                u = K_fisheye[0, 0] * x_f + cx
                v = K_fisheye[1, 1] * y_f + cy

                map_x[y, x] = u
                map_y[y, x] = v

        return map_x, map_y



    def fast_equirectangular_dewarping(self, frame, camera_id):
        """
        Applies precomputed remap for fast fisheye to equirectangular dewarping.

        Args:
            frame (np.ndarray): Input distorted fisheye image.
            camera_id (int): Camera selector (1 or 2) to choose corresponding remap.

        Returns:
            np.ndarray: Undistorted equirectangular image.
        """
        if camera_id == 1:
            undistorted_image = cv2.remap(frame, self.map_x1, self.map_y1,
                                        interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
        else:
            undistorted_image = cv2.remap(frame, self.map_x2, self.map_y2,
                                        interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)

        return undistorted_image

    
    def build_rectilinear_undistort_map(self, K_fisheye, D_fisheye, 
                                        output_size, fov_deg, 
                                        cx=None, cy=None):
        """
        Generates map_x, map_y to unwarp a fisheye image into a rectilinear (pinhole-like) image.

        Args:
            K_fisheye (np.ndarray): Intrinsic matrix of fisheye camera.
            D_fisheye (np.ndarray): Distortion coefficients (k1, k2, k3, k4).
            output_size (tuple): (width, height) of the desired undistorted image.
            fov_deg (tuple): (horizontal_FOV, vertical_FOV) of the output image in degrees.
            cx, cy: center of the distorted image (if None, taken from K).

        Returns:
            map_x, map_y: mapping matrices for cv2.remap()
        """
        width, height = output_size
        fov_x, fov_y = np.radians(fov_deg[0]), np.radians(fov_deg[1])
        
        # Intrinsics of the rectilinear output (pinhole model)
        fx = (width / 2) / np.tan(fov_x / 2)
        fy = (height / 2) / np.tan(fov_y / 2)
        cx_out = width / 2
        cy_out = height / 2

        # Source image center (from K)
        if cx is None:
            cx = K_fisheye[0, 2]
        if cy is None:
            cy = K_fisheye[1, 2]

        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        k1, k2, k3, k4 = D_fisheye.flatten()

        for y in range(height):
            for x in range(width):
                # Normalized coordinates in output rectilinear image
                x_r = (x - cx_out) / fx
                y_r = (y - cy_out) / fy

                r = np.sqrt(x_r ** 2 + y_r ** 2)
                if r == 0:
                    theta = 0
                else:
                    theta = np.arctan(r)

                # Fisheye distortion model (equidistant)
                theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

                if r != 0:
                    scale = theta_d / r
                    x_f = x_r * scale
                    y_f = y_r * scale
                else:
                    x_f = 0
                    y_f = 0

                # Project back to distorted image pixel coordinates
                u = K_fisheye[0, 0] * x_f + cx
                v = K_fisheye[1, 1] * y_f + cy

                map_x[y, x] = u
                map_y[y, x] = v


        return map_x, map_y  
    
    def dewarp_image(self,frame):

        undist = self.fast_equirectangular_dewarping(frame,1)

        return undist
        
        
    ############   STITCHING METHODS    #############

    def stitch_blend_optimized(self, left_img, right_img):
        """
        Stitches two dewarped images using intensity-based linear blending with gain correction.

        Args:
            left_img (np.ndarray): Left grayscale dewarped image.
            right_img (np.ndarray): Right grayscale dewarped image.

        Returns:
            np.ndarray: Final stitched image with smooth overlap.
        """
        assert left_img.shape == right_img.shape, f"Shape mismatch: {left_img.shape} vs {right_img.shape}"
        h, w = left_img.shape

        # Symmetric crop
        left_cropped = left_img[:, :w - self.trim_half]
        right_cropped = right_img[:, self.trim_half:]
        right_cropped = np.roll(right_cropped, self.vertical_correction, axis=0)

        # Gain correction (Brightness)
        mean_left = np.mean(left_cropped[:, -self.blend_px:])
        mean_right = np.mean(right_cropped[:, :self.blend_px])
        gain = mean_left / (mean_right + 1e-6)
        right_cropped = cv2.convertScaleAbs(right_cropped, alpha=gain)

        # Final canvas allocation
        stitched_w = left_cropped.shape[1] + right_cropped.shape[1] - self.blend_px
        stitched = np.zeros((h, stitched_w), dtype=np.uint8)
        stitched[:, :left_cropped.shape[1]] = left_cropped

        # Linear alpha blending
        left_patch = stitched[:, left_cropped.shape[1] - self.blend_px:left_cropped.shape[1]].astype(np.float32)
        right_patch = right_cropped[:, :self.blend_px].astype(np.float32)
        blended = ((1 - self.alpha) * left_patch + self.alpha * right_patch).astype(np.uint8)

        # Final stitch
        stitched[:, left_cropped.shape[1] - self.blend_px:left_cropped.shape[1]] = blended
        stitched[:, left_cropped.shape[1]:] = right_cropped[:, self.blend_px:]

        return stitched

        

