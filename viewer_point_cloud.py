#!/usr/bin/env python3
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtOpenGL import QGLWidget
import moderngl
from geometry_msgs.msg import PoseArray
from std_msgs.msg import UInt32
from nav_msgs.msg import Odometry, Path
from pyrr import Matrix44
import math
import scipy
import scipy.spatial
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import threading
from rcl_interfaces.msg import ParameterDescriptor

import os
# os.environ["ROS_DOMAIN_ID"] = "30"

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False, None  # This ray is parallel to this triangle.
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if not (0.0 <= u <= 1.0):
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if not (0.0 <= v <= 1.0):
        return False, None
    if u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        intersect_point = ray_origin + ray_direction * t
        return True, intersect_point
    else:
        return (
            False,
            None,
        )  # This means that there is a line intersection but not a ray intersection.


class Camera:
    def __init__(self):
        # 初始化相机参数
        self.eye = np.array([0.0, 0.0, 5.0])  # 相机位置
        self.target = np.array([0.0, 0.0, 0.0])  # 目标点
        self.up = np.array([0.0, 1.0, 0.0])  # 上方向
        self.update_view_matrix()  # 视图矩阵

        self.fov = 45.0
        self.proj_matrix = Matrix44.perspective_projection(self.fov, 1.0, 0.1, 100.0, dtype='f4')  # 投影矩阵
        
        self.last_mouse_pos = None  # 记录鼠标上一次的位置
        self.last_eye = None
        self.last_up = None
        self.move_speed = 0.1  # 相机移动速度

        self.width = None
        self.height = None

    def update_view_matrix(self):
        """更新视图矩阵"""
        self.view_matrix = Matrix44.look_at(self.eye, self.target, self.up, dtype='f4')

    def update_proj_matrix(self, width, height):
        """更新投影矩阵"""
        aspect_ratio = width / height
        self.proj_matrix = np.array(Matrix44.perspective_projection(self.fov, aspect_ratio, 0.1, 5000.0, dtype='f4'))
        self.width = width
        self.height = height

    def handle_mouse_press(self, pos):
        """处理鼠标按下事件"""
        self.last_mouse_pos = pos
        self.last_target = self.target
        self.last_eye = self.eye
        self.last_up = self.up/np.linalg.norm(self.up)

    def handle_mouse_release(self, pos):
        self.last_mouse_pos = None
        self.last_target = None
        self.last_eye = None
        self.last_up = None

    def handle_checkball_mouse_move(self, pos):
        """处理鼠标移动事件"""
        if self.last_mouse_pos is not None:
            x1 = self.last_mouse_pos.x()
            y1 = self.last_mouse_pos.y()
            x2 = pos.x()
            y2 = pos.y()

            if x1 == x2 and y1 == y2:
                return
            
            r = 0.5*min(self.width, self.height) 

            x1 = x1-self.width/2
            y1 = y1-self.height/2
            x2 = x2-self.width/2
            y2 = y2-self.height/2

            

            if x1*x1+y1*y1<=r*r/2:
                z1 = math.sqrt(r*r-x1*x1-y1*y1)
            else:
                z1 = r*r/2/(math.sqrt(x1*x1+y1*y1))
            
            if x2*x2+y2*y2<=r*r/2:
                z2 = math.sqrt(r*r-x2*x2-y2*y2)
            else:
                z2 = r*r/2/(math.sqrt(x2*x2+y2*y2))

            # 把屏幕空间坐标（x轴朝右，y轴朝下，z轴对着使用者）转化到世界坐标中
            forward = self.target-self.last_eye
            forward = forward/np.linalg.norm(forward)
            up = self.last_up
            right = np.cross(forward, up)

            v1 = x1*right+y1*(-up)+z1*(-forward)
            v2 = x2*right+y2*(-up)+z2*(-forward)
            v1 = v1/np.linalg.norm(v1)
            v2 = v2/np.linalg.norm(v2)
            N = np.cross(v1, v2)
            N = N/np.linalg.norm(N)
            theta = -math.acos(np.dot(v1, v2))
        
            rotation = scipy.spatial.transform.Rotation.from_rotvec(N*theta)

            forward = rotation.apply(forward)
            up = rotation.apply(up)

            # 正交化处理
            right = np.cross(forward, up)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            self.up = up
            self.eye = self.target-forward*np.linalg.norm(self.target-self.eye)
            self.update_view_matrix()
    
    def handle_drag_mouse_move(self, pos):
        """处理鼠标移动事件"""
        if self.last_mouse_pos is not None:
            x1 = self.last_mouse_pos.x()
            y1 = self.last_mouse_pos.y()
            x2 = pos.x()
            y2 = pos.y()
            
            forward = self.target-self.eye
            scale = np.linalg.norm(forward)
            up = self.up
            up = up/np.linalg.norm(up)
            right = np.cross(forward, up)
            right = right/np.linalg.norm(right)

            self.target = self.last_target+(x2-x1)*(-right)*0.001*scale+(y2-y1)*(up)*0.001*scale
            self.eye = self.last_eye+(x2-x1)*(-right)*0.001*scale+(y2-y1)*(up)*0.001*scale

            self.update_view_matrix()
            
            
    def zoom_in(self):
        # if self.fov>3:
        #     self.fov-=1
        # self.update_proj_matrix(self.width, self.height)
        if np.linalg.norm(self.eye-self.target)>0.2:
            distance = self.eye-self.target
            distance =  distance*0.98
            self.eye = self.target+distance
        self.update_view_matrix()
    
    def zoom_out(self):
        # if self.fov<140:
        #     self.fov+=1
        # self.update_proj_matrix(self.width, self.height)
        if np.linalg.norm(self.eye-self.target)<1000:
            distance = self.eye-self.target
            distance =  distance*1.0204
            self.eye = self.target+distance
        self.update_view_matrix()

    def move_forward(self):
        forward = self.target-self.eye
        forward = forward/np.linalg.norm(forward)
        self.target += forward*self.move_speed
        self.eye += forward*self.move_speed
        self.update_view_matrix()
    
    def move_backward(self):
        forward = self.target-self.eye
        forward = forward/np.linalg.norm(forward)
        self.target -= forward*self.move_speed
        self.eye -= forward*self.move_speed
        self.update_view_matrix()

    def move_left(self):
        forward = self.target-self.eye
        up = self.up
        right = np.cross(forward, up)
        right = right/np.linalg.norm(right)
        self.target -= right*self.move_speed
        self.eye -= right*self.move_speed
        self.update_view_matrix()

    def move_right(self):
        forward = self.target-self.eye
        up = self.up
        right = np.cross(forward, up)
        right = right/np.linalg.norm(right)
        self.target += right*self.move_speed
        self.eye += right*self.move_speed
        self.update_view_matrix()

    def move_up(self):
        up = self.up
        up = up/np.linalg.norm(up)
        self.target += up*self.move_speed
        self.eye += up*self.move_speed
        self.update_view_matrix()

    def move_down(self):
        up = self.up
        up = up/np.linalg.norm(up)
        self.target -= up*self.move_speed
        self.eye -= up*self.move_speed
        self.update_view_matrix()


class PointCloudWidget(QGLWidget):
    camera_selected_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # 确保键盘事件能够被捕获
        self.setMinimumHeight(400)
        self.setMinimumWidth(600)
        self.ctx = None
        self.program = None
        self.camera = Camera()  # 使用相机类
        self.camera.width = self.width()
        self.camera.height = self.height()
        
        # ROS2数据更新定时器
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.check_for_updates)
        self.ros_timer.start(16)  # 每16毫秒检查一次更新

        self.camera_list = []
        self.selected_camera = None
        self.block_list = []
        self.block_expa_list = []
        self.selected_block = None

        self.point_size = 1
        self.line_size = 3
        self.show_axis = True
        self.show_trackball = True
        self.show_scene_bbox = False
        self.camera_scale = 1.0

        self.c2ws = None

        self.line_color = np.array([58/255,134/255,255/255]).astype("f4")
        self.selected_line_color = np.array([255/255,0/255,110/255]).astype("f4")
        self.selected_point_color = np.array([251/255,86/255,7/255]).astype("f4")
        
        # 点云数据
        self.new_point_positions = np.empty((0, 3), dtype=np.float32)
        self.new_point_colors = np.empty((0, 3), dtype=np.float32)
        self.point_vbo_list = []
        self.point_vao_list = []
        
        # 数据锁
        self.data_lock = threading.Lock()
        self.data_updated = False

    def initializeGL(self):
        """初始化 OpenGL 环境"""
        self.ctx = moderngl.create_context()

        # 创建缓冲区和点云着色器
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_color;
        out vec3 frag_color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * model * vec4(in_position, 1.0);
            frag_color = in_color;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 frag_color;
        out vec4 frag_output;

        void main() {
            frag_output = vec4(frag_color, 1.0);
        }
        """

        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        axis_start = np.array([[-0.1, 0, 0], [0, -0.1, 0], [0, 0, -0.1]]).astype("f4")
        axis_end = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]]).astype("f4")
        axis_color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype('f4')
        axis = np.hstack([axis_start, axis_color, axis_end, axis_color])

        # 创建场景包围盒 VAO
        self.scene_bbox_vbo = None
        self.scene_bbox_vao = None

        # 坐标轴VAO
        self.axis_vbo = self.ctx.buffer(axis.tobytes())
        self.axis_vao = self.ctx.simple_vertex_array(self.program, self.axis_vbo, 'in_position', 'in_color')

        # 设置模型、视图和投影矩阵
        self.program['model'].write(Matrix44.identity(dtype='f4').tobytes())
        self.program['view'].write(self.camera.view_matrix.tobytes())
        self.program['projection'].write(self.camera.proj_matrix.tobytes())
    
    def update_pointcloud(self, points):
        """更新点云数据"""
        with self.data_lock:
            # 生成基于高度的颜色
            if len(points) > 0:
                colors = np.zeros_like(points)
                z_values = points[:, 2]  # 获取Z坐标
                if len(z_values) > 0 and not np.all(z_values == 0):
                    # 归一化Z值到0-1范围
                    z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-8)
                    # 将Z值映射为颜色 (蓝色到红色渐变)
                    colors[:, 0] = z_normalized  # 红色分量
                    colors[:, 2] = 1 - z_normalized  # 蓝色分量
                    colors[:, 1] = 0.3  # 绿色分量
                else:
                    # 默认颜色
                    colors = np.tile([0.5, 0.5, 1.0], (len(points), 1))
            else:
                colors = np.empty((0, 3), dtype=np.float32)
            
            self.new_point_positions = points.astype(np.float32)
            self.new_point_colors = colors.astype(np.float32)
            self.data_updated = True
            # print("update")

    def update_pointcloud_color(self, points, colors):
        with self.data_lock:
            # 生成基于高度的颜色
            self.new_point_positions = points.astype(np.float32)
            self.new_point_colors = colors.astype(np.float32)
            self.data_updated = True
            # print("update")

    def update_pointcloud_vao(self):
        """更新点云VAO"""
        # if self.point_vbo is not None:
        #     self.point_vbo.release()
        # if self.point_vao is not None:
        #     self.point_vao.release()
            
        if len(self.new_point_positions) > 0:
            point_vbo = self.ctx.buffer(np.hstack([self.new_point_positions, self.new_point_colors]).tobytes())
            point_vao = self.ctx.simple_vertex_array(self.program, point_vbo, 'in_position', 'in_color')
            
            self.point_vbo_list.append(point_vbo)
            self.point_vao_list.append(point_vao)

        # else:
        #     self.point_vbo = None
        #     self.point_vao = None

    def check_for_updates(self):
        """检查是否有新的点云数据需要更新"""
        with self.data_lock:
            if self.data_updated:
                self.update_pointcloud_vao()
                self.data_updated = False
                self.update()

    def paintGL(self):
        """绘制内容"""
        self.ctx.clear(0.1, 0.1, 0.1)  # 深灰色背景
        self.ctx.enable(moderngl.DEPTH_TEST)

        # 更新视图矩阵
        self.program['view'].write(self.camera.view_matrix.tobytes())
        self.program['projection'].write(self.camera.proj_matrix.tobytes())

        # 绘制点云
        for pcd in self.point_vao_list:
            
            self.ctx.point_size = self.point_size
            pcd.render(moderngl.POINTS)

        # 绘制线段
        for _, camera_vao in self.camera_list:
            self.ctx.line_width = self.line_size
            camera_vao.render(moderngl.LINES)

        for _, _, bbox_vao in self.block_list:
            self.ctx.line_width = self.line_size
            bbox_vao.render(moderngl.LINES)
        
        for _, _, bbox_vao in self.block_expa_list:
            self.ctx.line_width = self.line_size
            bbox_vao.render(moderngl.LINES)

        # 绘制坐标轴
        if self.show_axis:
            self.ctx.line_width = 2.0
            self.axis_vao.render(moderngl.LINES)

        if self.show_trackball:
            self.create_circles()
            self.ctx.line_width = 2.0
            # Note: circles rendering would need to be implemented similar to the original

        if self.show_scene_bbox:
            if self.scene_bbox_vao is not None:
                self.ctx.line_width = 1.0
                self.scene_bbox_vao.render(moderngl.LINES)

    def resizeGL(self, width, height):
        """调整视口大小"""
        self.ctx.viewport = (0, 0, width, height)
        self.program['projection'].write(self.camera.proj_matrix.tobytes())
        self.camera.update_proj_matrix(self.geometry().width(), self.geometry().height())
        
    def create_circles(self):
        """创建三个圆圈分别垂直于x、y、z轴"""
        # 简化版本，实际实现可参考pcd_viewer.py
        pass

    def mousePressEvent(self, event):
        """记录鼠标按下时的位置"""
        if event.button() == Qt.RightButton:
            # 简化处理
            pass
        else:
            self.camera.handle_mouse_press(event.pos())

    def mouseReleaseEvent(self, event):
        self.camera.handle_mouse_release(event.pos())

    def mouseMoveEvent(self, event):
        """处理鼠标拖动旋转视图"""
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.camera.handle_drag_mouse_move(event.pos())
        else:
            self.camera.handle_checkball_mouse_move(event.pos())
        self.update()

    def keyPressEvent(self, event):
        """处理键盘事件"""
        key = event.key()
        if key == QtCore.Qt.Key_W:
            self.camera.move_forward()  # W 键向前移动
        elif key == QtCore.Qt.Key_S:
            self.camera.move_backward()  # S 键向后移动
        elif key == QtCore.Qt.Key_A:
            self.camera.move_left()  # A 键向左移动
        elif key == QtCore.Qt.Key_D:
            self.camera.move_right()  # D 键向右移动
        self.update()  # 更新显示

    def wheelEvent(self, event):
        if event.angleDelta().y()>0:
            self.camera.zoom_in()
        else:
            self.camera.zoom_out()
        self.update()  # 更新显示


class PointCloudViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("PointCloud Control")
        self.setGeometry(100, 100, 800, 600)

        # 创建垂直布局
        layout = QVBoxLayout()
        font = QFont()
        font.setPointSize(4)  # 设置字体大小为9

        # 控制栏部分：水平布局
        control_layout = QHBoxLayout()
        
        # 点云大小
        self.point_size_label = QLabel("点云大小:")
        self.point_size_spinbox = QSpinBox(self)
        self.point_size_spinbox.setRange(1, 10)  # 设置范围
        self.point_size_spinbox.setSingleStep(1)  # 设置步长
        self.point_size_spinbox.setValue(1)  # 默认值为1
        self.point_size_spinbox.valueChanged.connect(self.change_point_size)

        # 线段粗细
        self.line_width_label = QLabel("线段粗细:")
        self.line_width_spinbox = QSpinBox(self)
        self.line_width_spinbox.setRange(1, 5)  # 设置范围
        self.line_width_spinbox.setSingleStep(1)  # 设置步长
        self.line_width_spinbox.setValue(3)  # 默认值为3
        self.line_width_spinbox.valueChanged.connect(self.change_line_size)

        # 相机大小
        self.camera_scale_label = QLabel("相机大小:")
        self.camera_scale_spinbox = QSpinBox(self)
        self.camera_scale_spinbox.setRange(1, 100)  # 设置范围
        self.camera_scale_spinbox.setSingleStep(1)  # 设置步长
        self.camera_scale_spinbox.setValue(1)  # 默认值为1
        self.camera_scale_spinbox.valueChanged.connect(self.change_camera_scale)

        # 是否显示坐标轴
        self.axis_checkbox = QCheckBox("坐标轴", self)
        self.axis_checkbox.setChecked(True)  # 默认勾选显示坐标轴
        self.axis_checkbox.toggled.connect(self.change_show_axis)

        # 是否显示轨迹球
        self.trackball_checkbox = QCheckBox("轨迹球", self)
        self.trackball_checkbox.setChecked(True)  # 默认勾选显示轨迹球
        self.trackball_checkbox.toggled.connect(self.change_show_trackball)

        # 将控件添加到控制栏
        control_layout.addWidget(self.point_size_label)
        control_layout.addWidget(self.point_size_spinbox)
        control_layout.addWidget(self.line_width_label)
        control_layout.addWidget(self.line_width_spinbox)
        control_layout.addWidget(self.camera_scale_label)
        control_layout.addWidget(self.camera_scale_spinbox)
        control_layout.addWidget(self.axis_checkbox)
        control_layout.addWidget(self.trackball_checkbox)

        control_layout.addStretch()

        # 点云显示区域：下部区域
        self.pointcloud_widget = PointCloudWidget()  # 用于展示点云

        # 将控制栏和显示区域添加到主布局
        layout.addLayout(control_layout)
        layout.addWidget(self.pointcloud_widget)

        # 设置窗口的主布局
        self.setLayout(layout)
    
    def change_point_size(self, value):
        self.pointcloud_widget.point_size = value
        self.pointcloud_widget.update()
        
    def change_line_size(self, value):
        self.pointcloud_widget.line_size = value
        self.pointcloud_widget.update()
        
    def change_camera_scale(self, value):
        self.pointcloud_widget.camera_scale = value
        self.pointcloud_widget.create_camera_geometry(self.pointcloud_widget.c2ws)
        self.pointcloud_widget.update()
        
    def change_show_axis(self, checked):
        self.pointcloud_widget.show_axis = checked
        self.pointcloud_widget.update()
        
    def change_show_trackball(self, checked):
        self.pointcloud_widget.show_trackball = checked
        self.pointcloud_widget.update()


class LIOVisualizer(Node):
    def __init__(self, point_cloud_widget):
        super().__init__('lio_receiver')
        self.point_cloud_widget = point_cloud_widget
        
        
        # 订阅所有FAST-LIVO发布的消息
        self.cloud_registered_sub = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_registered_callback,
            10)
            
        # self.cloud_registered_body_sub = self.create_subscription(
        #     PointCloud2,
        #     '/cloud_registered_body',
        #     self.cloud_registered_body_callback,
        #     10)
            
        # self.cloud_effected_sub = self.create_subscription(
        #     PointCloud2,
        #     '/cloud_effected',
        #     self.cloud_effected_callback,
        #     10)
            
        # self.laser_map_sub = self.create_subscription(
        #     PointCloud2,
        #     '/Laser_map',
        #     self.laser_map_callback,
        #     10)
            
        # self.odometry_sub = self.create_subscription(
        #     Odometry,
        #     '/Odometry',
        #     self.odometry_callback,
        #     10)
            
        # self.path_sub = self.create_subscription(
        #     Path,
        #     '/path',
        #     self.path_callback,
        #     10)
            
        self.get_logger().info('FAST-LIVO Listener 已启动，等待数据...')

    def cloud_registered_callback(self, msg):
        print("yes")
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
                colors.append([1, 0, 0])
            # colors.append([point[3]/255.0, point[4]/255.0, point[5]/255.0])
        
        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        colors = np.array(colors, dtype=np.float32).reshape(-1, 3)
        self.point_cloud_widget.update_pointcloud_color(points, colors)

    def cloud_registered_body_callback(self, msg):
        self.log_pointcloud_info('/cloud_registered_body', msg)

    def cloud_effected_callback(self, msg):
        self.log_pointcloud_info('/cloud_effected', msg)

    def laser_map_callback(self, msg):
        pass
        # print(np.frombuffer(msg.data, dtype=np.float32).shape)
        # points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8)[:, :3]
        # # colors = np.ones_like(points)
        # self.point_cloud_widget.update_pointcloud(points)
        # print(points.shape)

    def log_pointcloud_info(self, topic_name, msg):
        self.get_logger().info(f'收到 {topic_name} 点云数据:')
        self.get_logger().info(f'  帧ID: {msg.header.frame_id}')
        self.get_logger().info(f'  时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')
        self.get_logger().info(f'  点数: {msg.height * msg.width}')
        self.get_logger().info(f'  高度: {msg.height}')
        self.get_logger().info(f'  宽度: {msg.width}')

    def odometry_callback(self, msg):
        print(msg)
        self.get_logger().info('收到里程计数据:')
        self.get_logger().info(f'  帧ID: {msg.header.frame_id}')
        self.get_logger().info(f'  子帧ID: {msg.child_frame_id}')
        self.get_logger().info(f'  时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')
        
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.get_logger().info(f'  位置: x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f}')
        self.get_logger().info(f'  方向: x={orientation.x:.3f}, y={orientation.y:.3f}, z={orientation.z:.3f}, w={orientation.w:.3f}')

    def path_callback(self, msg):
        self.get_logger().info('收到路径数据:')
        self.get_logger().info(f'  帧ID: {msg.header.frame_id}')
        self.get_logger().info(f'  路径点数: {len(msg.poses)}')
        
        
    def cloud_callback(self, msg):
        """处理点云数据"""
        # 转换为Open3D格式
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8)[:, :3]
        # colors = np.ones_like(points)
        self.point_cloud_widget.update_pointcloud(points)
        print(points.shape)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        
        # with self.cloud_lock:
        #     self.latest_cloud = pcd
        #     self.cloud_updated = True

    def keyframe_callback(self, msg):
        """处理关键帧数据"""
        keyframe_count = msg.data
        print(f"Received keyframe count: {keyframe_count}")

    def pose_callback(self, msg):
        """处理位姿数据"""
        poses = []
        for pose in msg.poses:
            pos = pose.position
            ori = pose.orientation
            poses.append({
                'position': [pos.x, pos.y, pos.z],
                'orientation': [ori.x, ori.y, ori.z, ori.w]
            })
            
        with self.pose_lock:
            self.latest_poses = poses
            self.pose_updated = True


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS2 Point Cloud Viewer")
        self.setGeometry(100, 100, 800, 600)

        # 创建 OpenGL 小部件
        self.widget = PointCloudViewer(self)
        self.setCentralWidget(self.widget)


def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 创建Qt应用
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 创建ROS2节点
    node = LIOVisualizer(window.widget.pointcloud_widget)
    
    # 在单独的线程中运行ROS2
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()
    
    # 运行Qt应用
    exit_code = app.exec_()
    
    # 清理
    node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()