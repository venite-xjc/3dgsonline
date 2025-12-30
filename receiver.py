from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import UInt32
from nav_msgs.msg import Odometry, Path

from cv_bridge import CvBridge
import cv2
import numpy as np

import queue
import threading

from scipy.spatial.transform import Rotation


class FastLivo2Receiver(Node):
    def __init__(self, queue_max_length):
        super().__init__('fastlivo2_receiver')

        self.pcd_queue = queue.Queue(maxsize=queue_max_length)
        self.pcd_color_queue = queue.Queue(maxsize=queue_max_length)
        self.img_queue = queue.Queue(maxsize=queue_max_length)
        self.pose_queue = queue.Queue(maxsize=queue_max_length)
        self.K = None
        self.image_width = None
        self.image_height = None

        self.cloud_registered_sub = self.create_subscription(
            PointCloud2, 
            '/cloud_registered', 
            self.cloud_registered_callback, 
            50)
        
        self.image_registered_sub = self.create_subscription(
            Image, 
            '/rgb_img', 
            self.image_registered_callback, 
            50)
        self.bridge = CvBridge()

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            50
        )

        self.extrinsics_sub = self.create_subscription(
            PoseStamped,
            '/camera_extrinsics',
            self.extrinsics_callback,
            50
        )

    def extrinsics_callback(self, msg: PoseStamped):
        # 提取位置和朝向
        position = msg.pose.position
        orientation = msg.pose.orientation
        timestamp = msg.header.stamp
        # 从timestamp中提取sec和nanosec
        sec = timestamp.sec
        nanosec = timestamp.nanosec
        
        # # 打印时间戳
        # self.get_logger().info(f'Timestamp: sec={sec}, nanosec={nanosec}')
        
        # 将时间戳转换为浮点数（秒）
        # timestamp_float = sec + nanosec / 1e9

        # print("pose stamp: ", msg.header.stamp)
        qvec = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
        tvec = np.array([position.x, position.y, position.z])
        
        self.pose_queue.put([qvec, tvec, sec, nanosec])

    def camera_info_callback(self, msg: CameraInfo):
        # 提取内参矩阵 K
        K = msg.k  # 3x3 内参矩阵（展平为9元素列表）
        D = msg.d  # 畸变系数
        width = msg.width
        height = msg.height

        self.latest_camera_info = {
            'K': [[K[0], K[1], K[2]],
                  [K[3], K[4], K[5]],
                  [K[6], K[7], K[8]]],
            'D': D,
            'width': width,
            'height': height
        }
        self.K = np.array(K).reshape(3, 3)
        self.image_width = width
        self.image_height = height

        

    def cloud_registered_callback(self, msg):
        # print("yes")
        # 根据C++代码，点云包含x, y, z以及rgb颜色信息
        points_np = point_cloud2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        points = []
        colors = []
        for point in points_np:
            # 提取XYZ坐标
            points.append([point[0], point[1], point[2]])
            # 提取RGB颜色并归一化到0-1范围
            try:
                # 将 float32 的二进制位 reinterpret 为 uint32
                rgb_floats = np.array([point[3]], dtype=np.float32)

                rgb_uint32s = rgb_floats.view(np.uint32)  # 位 reinterpret，不改变内存
                r = (rgb_uint32s >> 16) & 0xFF
                g = (rgb_uint32s >> 8) & 0xFF
                b = rgb_uint32s & 0xFF
                colors.append([r/255.0, g/255.0, b/255.0])
                
            except:
                colors.append([0, 0, 0])
            # colors.append([point[3]/255.0, point[4]/255.0, point[5]/255.0])
        
        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        colors = np.array(colors, dtype=np.float32).reshape(-1, 3)

        self.pcd_queue.put([points, colors])

    def image_registered_callback(self, msg):
        # print("image stamp: ", msg.header.stamp)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        np_image = np.array(cv_image)/255.0
        timestamp = msg.header.stamp
        sec = timestamp.sec
        nanosec = timestamp.nanosec
        self.img_queue.put([np_image, sec, nanosec])




if __name__ == '__main__':
    import rclpy
    rclpy.init()
    node = FastLivo2Receiver(queue_max_length=10)

    # 启动 ROS spin 线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        while rclpy.ok():
            try:
                pcd = node.pcd_queue.get(timeout=0.01)
                image = node.img_queue.get(timeout=0.01)
                pose = node.pose_queue.get(timeout=0.01)
                # 👉 在这里处理点云：可视化、保存、推理等
                print(f"Processing point cloud with {len(pcd[0])} points")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()