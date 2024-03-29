import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

def hat(k):
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2

def euler_from_quaternion(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return [roll, pitch, yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.obj_pose = None

        # Declare parameters
        self.declare_parameter('world_frame_id', 'odom')
        self.declare_parameter('Kp_linear', 0.5)
        self.declare_parameter('Kp_angular', 0.5)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        self.sub_detected_obj_pose = self.create_subscription(PoseStamped, '/detected_color_object_pose', self.detected_obj_pose_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obj_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if np.linalg.norm(center_points[:2]) > 3 or center_points[2] > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            return

        self.obj_pose = cp_world

    def get_current_object_pose(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x, robot_world_y, robot_world_z = transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            object_pose = robot_world_R @ self.obj_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
        except TransformException as e:
            return None

        return object_pose

    def timer_update(self):
        if self.obj_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return

        current_object_pose = self.get_current_object_pose()
        if current_object_pose is not None:
            distance = np.linalg.norm(current_object_pose[:2])
            angle_to_target = math.atan2(current_object_pose[1], current_object_pose[0])
            cmd_vel = self.controller(distance, angle_to_target)
        else:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, distance, angle):
        Kp_linear = self.get_parameter('Kp_linear').get_parameter_value().double_value
        Kp_angular = self.get_parameter('Kp_angular').get_parameter_value().double_value

        cmd_vel = Twist()
        stop_distance = 0.1

        if distance > stop_distance:
            cmd_vel.linear.x = min(Kp_linear * (distance - stop_distance), 0.5)
        else:
            cmd_vel.linear.x = 0.0

        cmd_vel.angular.z = Kp_angular * angle

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
