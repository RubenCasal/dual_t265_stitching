import cv2
import numpy as np

class DualFisheyeStitcher:
    def __init__(self, fov, frame_width, frame_height,K_cam1, D_cam1, fov_h1, fov_v1, K_cam2, D_cam2, fov_h2, fov_v2):
    
        self.output_size = (1000,1000)
        self.dewarped_width = 1000
        self.fov = 163
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

        # Calculate overlap percentage
        self.overlap_percentage = self.calculate_overlap_percentage()

        
        # Precompute LUT for camera1
        self.map_x1, self.map_y1 = self.build_stereographic_undistort_map_soft_fov(self.K_cam1, self.D_cam1,self.output_size, self.fov1)

        # Precompute LUT for camera2
        self.map_x2, self.map_y2 = self.build_stereographic_undistort_map_soft_fov(self.K_cam2, self.D_cam2,self.output_size, self.fov2)

        # Rotate 180 degrees one of the cameras
        self.map_x1 = cv2.flip(self.map_x1, -1)
        self.map_y1 = cv2.flip(self.map_y1, -1)



    def compute_required_camera_angle(self, desired_overlap_percentage = 0.1):

        desired_overlap_percentage = np.clip(desired_overlap_percentage, 0.0, 1)


        # maximum angle that can be convered without overlap
        max_possible_fov = self.fovh1_eff + self.fovh2_eff

        # overlap in degrees
        overlap_angle = (desired_overlap_percentage * max_possible_fov) / (1-desired_overlap_percentage)

        # angle between cameras
        angle = max_possible_fov- overlap_angle

        return angle, 

    def calculate_overlap_percentage(self):
        self.fovh1_eff = 121
        self.fovh2_eff = 111
        # Compute angular coverage range of both cameras
        cam1_min = -self.fovh1_eff /2 
        cam1_max =  self.fovh1_eff /2


        cam2_center = 90
        cam2_min = cam2_center - self.fovh2_eff /2
        cam2_max = cam2_center + self.fovh2_eff / 2

        # Compute overlap angular width
        overlap_min = max(cam1_min, cam2_min)
        overlap_max = min(cam1_max, cam2_max)
        overlap_angle = max(0.0, overlap_max - overlap_min)

        # Normalize to percentage of image width (based on each camera FOV)
        overlap_pct_cam1 = overlap_angle / self.fovh1_eff if self.fovh1_eff > 0 else 0.0
        overlap_pct_cam2 = overlap_angle / self.fovh2_eff if self.fovh2_eff > 0 else 0.0

        # Return average overlap to apply symmetrical shift
        return (overlap_pct_cam1 + overlap_pct_cam2) / 2.0

    def calculate_overlapping_region(self):
        # Assuming the 2 cameras are placed 180 degrees from each other

        # Calculate the overlap angle
        overlap_angle = 2 * self.fov - 360

        # Convert to overlap percentage
        overlap_percentage = overlap_angle / (2 * self.fov) # Corrected calculation

        return overlap_percentage # Return the calculated percentage

    def stitch_by_shift(self, left_frame, right_frame,overlap_percentage):
        # Overlapping in the horizontal section
        overlap = int(self.dewarped_width * overlap_percentage)

        # Crop frames
        left_main = left_frame[:, :self.dewarped_width - overlap]
        right_main = right_frame[:, overlap:]
        left_overlap = left_frame[:, self.dewarped_width - overlap:]
        right_overlap = right_frame[:, :overlap]

        # merge in the overlapping zone
        alpha = np.linspace(1, 0, overlap).reshape(1, -1)
        blended = (left_overlap * alpha + right_overlap * (1 - alpha)).astype(np.uint8)
        blended = cv2.GaussianBlur(blended, (3, 3), sigmaX=0.8)


        # Concatenate frames
        result = np.concatenate([left_main, blended, right_main], axis=1)

        return result
    
    def build_stereographic_undistort_map(self, K_fisheye, D_fisheye, output_size, fov_deg=(90, 60), cx=None, cy=None):
        """
        Generates map_x, map_y to unwarp a fisheye image into a stereographic projection.

        Args:
            K_fisheye (np.ndarray): Intrinsic matrix of fisheye camera.
            D_fisheye (np.ndarray): Distortion coefficients (k1, k2, k3, k4).
            output_size (tuple): (width, height) of the desired undistorted image.
            fov_deg (tuple): (horizontal_FOV, vertical_FOV) in degrees.
            cx, cy: center of the distorted image (if None, taken from K).

        Returns:
            map_x, map_y: mapping matrices for cv2.remap()
        """
        width, height = output_size
        fov_x, fov_y = np.radians(fov_deg[0]), np.radians(fov_deg[1])

        # Output image intrinsics for stereographic model (fx, fy based on FOV)
        fx = (width / 2) / np.tan(fov_x / 2)
        fy = (height / 2) / np.tan(fov_y / 2)
        cx_out = width / 2
        cy_out = height / 2

        # Set default principal point if not given
        cx = float(K_fisheye[0, 2]) if cx is None else float(cx)
        cy = float(K_fisheye[1, 2]) if cy is None else float(cy)

        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        k1, k2, k3, k4 = [float(k) for k in D_fisheye.flatten()]

        for y in range(height):
            for x in range(width):
                # Normalize coordinates in output image
                x_r = (x - cx_out) / fx
                y_r = (y - cy_out) / fy

                r = np.sqrt(x_r ** 2 + y_r ** 2)

                if r == 0:
                    theta = 0.0
                else:
                    theta = 2 * np.arctan(r / 2)  # stereographic model

                theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

                if r != 0:
                    scale = theta_d / r
                    x_f = x_r * scale
                    y_f = y_r * scale
                else:
                    x_f = 0.0
                    y_f = 0.0

                u = float(K_fisheye[0, 0]) * x_f + cx
                v = float(K_fisheye[1, 1]) * y_f + cy

                map_x[y, x] = float(u)
                map_y[y, x] = float(v)

        return map_x, map_y

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
    
    def build_stereographic_undistort_map_soft_fov(self,
    K_fisheye,
    D_fisheye,
    output_size,
    fov_deg=(90, 60),
    fov_margin=0.9,  # 986 del FOV original
    cx=None,
    cy=None
):
        """
        Generates stereographic undistortion map with slightly reduced FOV to avoid black borders.

        Args:
            K_fisheye (np.ndarray): Intrinsic matrix of fisheye camera.
            D_fisheye (np.ndarray): Distortion coefficients (k1, k2, k3, k4).
            output_size (tuple): (width, height) of the desired output image.
            fov_deg (tuple): (horizontal_FOV, vertical_FOV) in degrees.
            fov_margin (float): ratio to reduce FOV (e.g., 0.98 = 98% of original).
            cx, cy: optional center of projection (default from intrinsics).

        Returns:
            map_x, map_y: mapping matrices for cv2.remap()
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
        if (camera_id == 1):
            undistorted_image = cv2.remap(frame, self.map_x1, self.map_y1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            undistorted_image = cv2.remap(frame, self.map_x2, self.map_y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            

        return undistorted_image



    def stitch_frames(self, left_frame, right_frame):
        #  Fast dewarp with LUTs (Parallelized)
        
        undist_left = self.fast_equirectangular_dewarping(left_frame, 1)
        undist_right = self.fast_equirectangular_dewarping(right_frame, 2)
        
        # stich undistorted images
        stitched = self.stitch_by_shift(undist_left, undist_right)
     
        return stitched
    
    def dewarp_image(self,frame):

        undist = self.fast_equirectangular_dewarping(frame,1)

        return undist
        
        
        
        
        
        
        _