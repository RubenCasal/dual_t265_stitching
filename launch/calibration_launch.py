import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package="dual_t265_stitching",
            executable = "dual_t265_node",
            name= "dual_t265_node",
            output="screen"
        ),

    launch_ros.actions.Node(
                package='dual_t265_stitching',
                executable='overlap_calibration.py',
                name='overlap_calibration',
                output='screen',
                parameters=[],
                arguments=[os.path.join(os.getenv('ROS_WS', '/home/rcasal/ros2_ws'), 'install/dual_t265_stitching/lib/dual_t265_stitching/overlap_calibration.py')]
            )

    ])
        