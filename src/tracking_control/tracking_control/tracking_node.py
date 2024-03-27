import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obj_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_obj_pose = self.create_subscription(PoseStamped, '/detected_color_object_pose', self.detected_obj_pose_callback, 10)
    
        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obj_pose_callback(self, msg):
        # Retrieve the world frame ID parameter
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Extract the center point coordinates of the detected object from the message
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    
        # Filtering based on distance and height
        # You can adjust or remove these filters based on your application's needs
        if np.linalg.norm(center_points[:2]) > 3 or center_points[2] > 0.7:
            # If the object is too far or too high, ignore this detection
            return
    
        try:
            # Look up the transformation from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            # Convert the quaternion to a rotation matrix
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            # Apply the rotation and translation to get the object's pose in the world frame
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            # If there is an error in transforming the pose, log the error
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Update the global object pose with the transformed pose
        self.obj_pose = cp_world

        
    def get_current_object_pose(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            object_pose = robot_world_R@self.obj_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return
        
        return object_pose
   
    def timer_update(self):
    if self.obj_pose is None:
        # If no object is detected, stop the robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.pub_control_cmd.publish(cmd_vel)
        return
    
    # Try to get the current object pose relative to the robot's base_footprint frame
    current_object_pose = self.get_current_object_pose()
    
    if current_object_pose is not None:
        # Calculate the distance and angle to the target object
        distance = np.linalg.norm(current_object_pose[:2])  # Consider only X and Y for distance
        angle_to_target = math.atan2(current_object_pose[1], current_object_pose[0])  # Angle in the XY plane
        
        # Control strategy: Calculate velocities based on distance and angle
        cmd_vel = self.controller(distance, angle_to_target)
    else:
        # If we can't get the current pose for some reason, stop the robot as a precaution
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
    
    # Publish the velocity command
    self.pub_control_cmd.publish(cmd_vel)
    #################################################
    
    def controller(self, distance, angle):
    # Proportional control gains
    Kp_linear = 0.5  # Gain for linear velocity control
    Kp_angular = 1.0  # Gain for angular velocity control

    # Initialize the Twist message
    cmd_vel = Twist()

    # Stop 0.3 meters away from the target
    stop_distance = 0.3

    # Control law for linear velocity: Proportional control to slow down as the robot approaches the target
    if distance > stop_distance:
        # Cap the linear speed to a maximum value, for example, 0.5 m/s
        cmd_vel.linear.x = min(Kp_linear * (distance - stop_distance), 0.5)
    else:
        # Stop if within the stopping distance
        cmd_vel.linear.x = 0.0

    # Control law for angular velocity: Proportional control to align the robot towards the target
    # The angular velocity is determined by the angle to the target, scaled by a proportional gain
    cmd_vel.angular.z = Kp_angular * angle

    # Return the computed command velocities
    return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
