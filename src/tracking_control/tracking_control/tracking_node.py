import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector.
    """
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix.
    """
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2

def euler_from_quaternion(q):
    """
    Converts a quaternion to Euler angles.
    """
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return [roll, pitch, yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        self.obj_pose = None  # Current object pose

        self.declare_parameter('world_frame_id', 'odom')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        self.sub_detected_obj_pose = self.create_subscription(PoseStamped, '/detected_color_object_pose', self.detected_obj_pose_callback, 10)
    
        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obj_pose_callback(self, msg):
        self.get_logger().info('detected_obj_pose_callback invoked')  # Debug point
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if np.linalg.norm(center_points[:2]) > 3 or center_points[2] > 0.7:
            self.get_logger().info('Ignoring detected object: too far or too high')  # Debug point
            return

        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
            self.get_logger().info(f'Object world pose: {cp_world}')  # Debug point
        except TransformException as e:
            self.get_logger().error(f'Transform Error: {e}')  # Debug point
            return

        self.obj_pose = cp_world

    def get_current_object_pose(self):
        self.get_logger().info('get_current_object_pose invoked')  # Debug point
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x, robot_world_y, robot_world_z = transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            object_pose = robot_world_R @ self.obj_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
            self.get_logger().info(f'Current object pose: {object_pose}')  # Debug point
        except TransformException as e:
            self.get_logger().error(f'Transform error: {e}')  # Debug point
            return None

        return object_pose

    def timer_update(self):
        self.get_logger().info('timer_update invoked')  # Debug point
        if self.obj_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            self.get_logger().info('No object detected, stopping robot')  # Debug point
            return

        current_object_pose = self.get_current_object_pose()
        if current_object_pose is not None:
            distance = np.linalg.norm(current_object_pose[:2])
            angle_to_target = math.atan2(current_object_pose[1], current_object_pose[0])
            cmd_vel = self.controller(distance, angle_to_target)
            self.get_logger().info(f'Publishing cmd_vel: Linear={cmd_vel.linear.x}, Angular={cmd_vel.angular.z}')  # Debug point
        else:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.get_logger().error('Failed to obtain current object pose, stopping robot')  # Added debug point for the else statement

        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, distance, angle):
        self.get_logger().info(f'controller invoked with distance: {distance}, angle: {angle}')  # Debug point
        Kp_linear = 0.5
        Kp_angular = 1.0

        cmd_vel = Twist()
        stop_distance = 0.1

        if distance > stop_distance:
            cmd_vel.linear.x = min(Kp_linear * (distance - stop_distance), 0.5)
            self.get_logger().info(f'Moving towards target: Linear velocity set to {cmd_vel.linear.x}')  # Debug point for movement
        else:
            cmd_vel.linear.x = 0.0
            self.get_logger().info('Within stop distance: Stopping robot')  # Debug point for stopping
        cmd_vel.angular.z = Kp_angular * angle
        self.get_logger().info(f'Angular velocity set to {cmd_vel.angular.z}')  # Debug point for angular velocity

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
