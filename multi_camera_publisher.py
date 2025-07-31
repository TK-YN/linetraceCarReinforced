import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2

class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher')
        self.bridge = CvBridge()
        self.camera_topics = [
            '/camera_1/image_raw',
            '/camera_2/image_raw',
            '/camera_3/image_raw',
            '/camera_4/image_raw',
            '/camera_5/image_raw',
            '/camera_6/image_raw',
            '/camera_7/image_raw',
            '/camera_8/image_raw'
        ]
        self.brightness_values = [0.0] * len(self.camera_topics)

        self.my_subscriptions = []
        for i, topic in enumerate(self.camera_topics):
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, idx=i: self.image_callback(msg, idx),
                10
            )
            self.my_subscriptions.append(sub)

        #輝度配列を強化学習用に送るPublisher
        self.brightness_pub = self.create_publisher(Float64MultiArray, '/brightness_array', 10)

        #定期的に配列を送信
        self.timer = self.create_timer(0.1, self.publish_brightness_array)

    def image_callback(self, msg, idx):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        self.brightness_values[idx] = brightness
        #self.get_logger().info(f'Camera {idx+1}: {brightness:.2f}')

    def publish_brightness_array(self):
        msg = Float64MultiArray()
        msg.data = self.brightness_values
        self.brightness_pub.publish(msg)
        #self.get_logger().info(f'Published brightness array: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    
    node = MultiCameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        node.stop_wheels()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
