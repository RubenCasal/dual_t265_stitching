#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <librealsense2/rs.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>


class T265Node : public rclcpp::Node
{
public:
    T265Node(const std::string& serial, const std::string& prefix) 
    : Node((prefix +"_t265_node").c_str()), prefix_(prefix), tf_broadcaster_(this)
    {
        RCLCPP_INFO(this->get_logger(), "Starting T265 node with prefix: %s", prefix.c_str());

        // Configure the camera pipeline
        cfg_.enable_device(serial);
        cfg_.enable_stream(RS2_STREAM_ACCEL);
        cfg_.enable_stream(RS2_STREAM_GYRO);
        cfg_.enable_stream(RS2_STREAM_POSE);
        cfg_.enable_stream(RS2_STREAM_FISHEYE, 1);
        cfg_.enable_stream(RS2_STREAM_FISHEYE, 2);
        pipe_.start(cfg_);

        // Create publishers
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(prefix_ +"/odom", 10);
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(prefix_ + "/imu", 10);
        fisheye_left_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(prefix_ + "/fisheye_left", 10);
        fisheye_right_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(prefix_ +"/fisheye_right", 10);

        // Timer to periodically publish data
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&T265Node::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "T265 node started successfully.");
    }

private:
    void timer_callback()
    {
        auto frames = pipe_.wait_for_frames();

        // Get IMU data
        if (auto gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
        {
            auto gyro_data = gyro_frame.as<rs2::motion_frame>().get_motion_data();
            imu_msg_.angular_velocity.x = gyro_data.x;
            imu_msg_.angular_velocity.y = gyro_data.y;
            imu_msg_.angular_velocity.z = gyro_data.z;
        }

        if (auto accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
        {
            auto accel_data = accel_frame.as<rs2::motion_frame>().get_motion_data();
            imu_msg_.header.stamp = this->now();
            imu_msg_.header.frame_id = prefix_ +"_imu_frame";
            imu_msg_.linear_acceleration.x = accel_data.x;
            imu_msg_.linear_acceleration.y = accel_data.y;
            imu_msg_.linear_acceleration.z = accel_data.z;

            imu_publisher_->publish(imu_msg_);
        }

        // Get pose data
        if (auto pose_frame = frames.first_or_default(RS2_STREAM_POSE))
        {
            auto pose_data = pose_frame.as<rs2::pose_frame>().get_pose_data();
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = this->now();
            odom_msg.header.frame_id = "odom";
            odom_msg.child_frame_id = "t265_frame";

            // Position
            odom_msg.pose.pose.position.x = -pose_data.translation.z;
            odom_msg.pose.pose.position.y = -pose_data.translation.x;
            odom_msg.pose.pose.position.z = pose_data.translation.y;

            // Orientation
            odom_msg.pose.pose.orientation.x = -pose_data.rotation.z;
            odom_msg.pose.pose.orientation.y = -pose_data.rotation.x;
            odom_msg.pose.pose.orientation.z = pose_data.rotation.y;
            odom_msg.pose.pose.orientation.w = pose_data.rotation.w;

            odom_publisher_->publish(odom_msg);

            // PUBLISH TF TRANSFORM
            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped.header.stamp = odom_msg.header.stamp;
            transform_stamped.header.frame_id = "odom";
            transform_stamped.child_frame_id = "t265_frame";

            transform_stamped.transform.translation.x = odom_msg.pose.pose.position.x;
            transform_stamped.transform.translation.y = odom_msg.pose.pose.position.y;
            transform_stamped.transform.translation.z = odom_msg.pose.pose.position.z;

            transform_stamped.transform.rotation = odom_msg.pose.pose.orientation;

            tf_broadcaster_.sendTransform(transform_stamped);
        }

        // Get and publish fisheye left image
        if (auto fisheye_left_frame = frames.first_or_default(RS2_STREAM_FISHEYE, RS2_FORMAT_ANY))
        {
            auto frame_data = fisheye_left_frame.as<rs2::video_frame>();
            cv::Mat image(cv::Size(frame_data.get_width(), frame_data.get_height()), CV_8UC1,
                          (void *)frame_data.get_data(), cv::Mat::AUTO_STEP);

            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "t265_fisheye_left";

            sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "mono8", image).toImageMsg();
            fisheye_left_publisher_->publish(*msg);
        }

        // Get and publish fisheye right image
        if (auto fisheye_right_frame = frames.first_or_default(RS2_STREAM_FISHEYE, RS2_FORMAT_ANY))
        {
            auto frame_data = fisheye_right_frame.as<rs2::video_frame>();
            cv::Mat image(cv::Size(frame_data.get_width(), frame_data.get_height()), CV_8UC1,
                          (void *)frame_data.get_data(), cv::Mat::AUTO_STEP);

            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "t265_fisheye_right";

            sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "mono8", image).toImageMsg();
            fisheye_right_publisher_->publish(*msg);
        }
    }

    // prefix of the used camera
    std::string prefix_;

    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr fisheye_left_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr fisheye_right_publisher_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // RealSense pipeline
    rs2::pipeline pipe_;
    rs2::config cfg_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    sensor_msgs::msg::Imu imu_msg_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node1 = std::make_shared<T265Node>("943222110291","camera1");
    auto node2 = std::make_shared<T265Node>("201222111352","camera2");

    rclcpp::executors::MultiThreadedExecutor exec;


    exec.add_node(node1);
    exec.add_node(node2);
    exec.spin();


    rclcpp::shutdown();
    return 0;
}