import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import sys

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

class ColorObjDetectionNode(Node):
    def __init__(self):
        super().__init__('color_obj_detection_node')
        self.get_logger().info('Initializing Color Object Detection Node')

        # Logging parameters
        color_low = self.declare_parameter('color_low', [110, 50, 150]).get_parameter_value().integer_array_value
        color_high = self.declare_parameter('color_high', [130, 255, 255]).get_parameter_value().integer_array_value
        object_size_min = self.declare_parameter('object_size_min', 1000).get_parameter_value().integer_value
        self.get_logger().info(f'Set color_low to {color_low}, color_high to {color_high}, object_size_min to {object_size_min}')

        # Set up CV bridge
        self.br = CvBridge()
        self.get_logger().info('CV Bridge initialized')

        # Initialize transform listener and buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('Transform listener and buffer initialized')

        # Setup publisher and subscriber
        self.setup_pub_sub()

        # Setup time synchronizer
        self.setup_time_synchronizer()

    def setup_pub_sub(self):
        self.pub_detected_obj = self.create_publisher(Image, '/detected_color_object', 10)
        self.pub_detected_obj_pose = self.create_publisher(PoseStamped, '/detected_color_object_pose', 10)
        self.sub_rgb = Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth/points')
        self.get_logger().info('Publishers and subscribers initialized')

    def setup_time_synchronizer(self):
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.camera_callback)
        self.get_logger().info('Time synchronizer setup completed')

    def camera_callback(self, rgb_msg, points_msg):
    self.get_logger().info('Received synchronized RGB and Depth messages')

    # Get parameters and prepare images
    color_low = np.array(self.get_parameter('color_low').value)
    color_high = np.array(self.get_parameter('color_high').value)
    object_size_min = self.get_parameter('object_size_min').value
    rgb_image = self.br.imgmsg_to_cv2(rgb_msg, "bgr8")
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Process image for color detection
    color_mask = cv2.inRange(hsv_image, color_low, color_high)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    self.get_logger().info(f'Found {len(contours)} contours')

    if not contours:
        self.get_logger().info('No contours found')
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w * h < object_size_min:
        self.get_logger().info('Ignoring detected object: too small or out of size range')
        return

    self.get_logger().info(f'Drawing bounding box at {x}, {y}, width {w}, height {h}')
    cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    try:
        pointid = (y * points_msg.row_step) + (x * points_msg.point_step)
        if pointid >= len(points_msg.data):
            self.get_logger().error('Computed point index exceeds point cloud data buffer size')
            return

        (X, Y, Z) = struct.unpack_from('fff', points_msg.data, offset=pointid)
        self.get_logger().info(f'Object position in camera frame: X={X}, Y={Y}, Z={Z}')
    except struct.error as e:
        self.get_logger().error('Failed to extract depth data due to incorrect struct unpacking: {}'.format(e))
        return
    except Exception as e:
        self.get_logger().error('Unexpected error occurred while extracting depth data: {}'.format(e))
        return

    detected_obj_pose = PoseStamped()
    detected_obj_pose.header.frame_id = 'base_footprint'
    detected_obj_pose.header.stamp = rgb_msg.header.stamp
    detected_obj_pose.pose.position.x = X
    detected_obj_pose.pose.position.y = Y
    detected_obj_pose.pose.position.z = Z
    self.pub_detected_obj_pose.publish(detected_obj_pose)
    self.pub_detected_obj.publish(self.br.cv2_to_imgmsg(rgb_image, encoding='bgr8'))
    self.get_logger().info('Published detected object and pose')

def main(args=None):
    rclpy.init(args=args)
    node = ColorObjDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
