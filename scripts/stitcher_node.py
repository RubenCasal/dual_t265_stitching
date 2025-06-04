#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data
from threading import Thread
from queue import Queue
from dual_fisheye_stitcher import DualFisheyeStitcher


class StitchingNode(Node):
    def __init__(self):
        super().__init__("stitching_node")
        self.bridge = CvBridge()

        self.image_width = 848
        self.image_height = 800
        self.image_fov = 163

        # Calibration patameters camera 1
 
        serial_cam1 = "943222110291"

        K_cam1 = np.array([
            [285.720703125, 0, 411.068206787109],
            [0, 285.427490234375, 394.509887695312],
            [0, 0, 1]
        ], dtype=np.float32)

        D_cam1 = np.array([
            -0.0046993619762361,
            0.0400081910192966,
            -0.037823498249054,
            0.00574744818732142,
        ], dtype=np.float32)

        fov_h1 = 121
        fov_v1 = 109

        # Calibration parameters camera 2

        serial_cam2 = "201222111352"

        K_cam2 = np.array([
            [286.497100830078, 0, 421.204895019531],
            [0, 286.372497558594, 394.644195556641],
            [0, 0, 1]
        ], dtype=np.float32)

        D_cam2 = np.array([
            -0.0124582499265671,
            0.0536976382136345,
            -0.0504140295088291,
            0.0101647302508354
        ], dtype=np.float32)


        fov_h2 = 111.9
        fov_v2 = 108.8

        # Initialise the stitching class
        self.get_logger().info("Calculating LUTs for dewarping ")

        self.dual_stitcher = DualFisheyeStitcher(
            frame_width= self.image_width,
            frame_height= self.image_height,
            K_cam1 = K_cam1,
            D_cam1 = D_cam1,
            fov_h1 = fov_h1,
            fov_v1 = fov_v1,
            K_cam2 = K_cam2,
            D_cam2 = D_cam2,
            fov_h2 = fov_h2,
            fov_v2 = fov_v2,
            overlaping_region=0.135,
            blending_ratio = 0.1,
            vertical_correction=1
        )

        # Subscribe to fisheye cameras topics
        self.subscription_left_camera = self.create_subscription(
            Image,
            "/camera2/fisheye_left",
            self.image_callback_left,
            qos_profile_sensor_data
        )
    
        self.subscription_right_camera = self.create_subscription(
            Image,
            "/camera1/fisheye_left",
            self.image_callback_right,
            qos_profile_sensor_data
        )
        
        # Publisher for annotated images
        self.publisher_dewarped_left = self.create_publisher(Image, "/dewarped_left", 1)
        self.publisher_dewarped_right = self.create_publisher(Image, "/dewarped_right", 1)


        self.last_left = None # Left camera frame
        self.last_right = None # Right camera frame


        # Publisher for the final stitched image
        self.publisher_stitched = self.create_publisher(Image, "/stitched_image",1)


        self.get_logger().info("✅ Stitiching node initialized correctly")
        self.get_logger().info(f"Stitching has an overlap of {self.dual_stitcher.overlaping_region}")



    def image_callback_left(self, msg):
        try:

            cv_image = self.bridge.imgmsg_to_cv2(msg)

            undistorted = self.dual_stitcher.fast_equirectangular_dewarping(cv_image,1)
            self.last_left = undistorted
         
            ros_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="mono8")
            self.publisher_dewarped_left.publish(ros_msg)


            self.dual_stitching()
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to process image: {e}")


    def image_callback_right(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            undistorted = self.dual_stitcher.fast_equirectangular_dewarping(cv_image,2)
            self.last_right = undistorted

            ros_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="mono8")
            self.publisher_dewarped_right.publish(ros_msg)


            self.dual_stitching()

        except Exception as e:
            self.get_logger().error(f"❌ Failed to process image: {e}")

    def dual_stitching(self):
        if self.last_left is None or self.last_right is None:
            return

        try:
            # Save raw images for offline testing/debugging
            cv2.imwrite("/home/rcasal/ros2_ws/src/dual_t265_stitching/final_left.png", self.last_left)
            cv2.imwrite("/home/rcasal/ros2_ws/src/dual_t265_stitching/final_right.png", self.last_right)

            stiched_image = self.dual_stitcher.stitch_blend_optimized(self.last_left, self.last_right)
    
            ros_msg = self.bridge.cv2_to_imgmsg(stiched_image, encoding="mono8")
            self.publisher_stitched.publish(ros_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Stitching failed: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = StitchingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()