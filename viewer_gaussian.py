import sys
from contextlib import contextmanager
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.Qt import QPixmap, QPoint, Qt, QPainter, QIcon
import numpy as np
import time
# from glumpy import app as glumpy_app, gl, gloo
# from glumpy.app.window import key as Key
import torch
# CUDA_ENABLE_GL = True
# from pycuda.gl import graphics_map_flags
# import pycuda.driver
# sys.path.append('/home/xjc/code/LargePGSR/PGSR')
# from scene import Scene
# from gaussian_renderer import render
# from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, focal2fov, fov2focal
# from argparse import ArgumentParser
# from torchvision.utils import save_image
# from arguments import ModelParams, PipelineParams, get_combined_args
# from scene.gaussian_model import GaussianModel
# from torchvision.io import read_image
# import random
# import copy
# import os
# import json
# import moderngl
import torch.nn as nn
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
# from scene.cameras import OnlineCam


class RenderCam():
    @torch.no_grad()
    def __init__(self, image_width=640, image_height=512, FoVx = 1, FoVy=0.75):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy


        self.R = -torch.eye(3).cuda()
        self.T = torch.zeros(3).cuda()

        # world_view_transform是W2C.T
        self.world_view_transform = torch.eye(4).cuda()
        self.world_view_transform[:3, :3] = self.R
        self.world_view_transform[:3, 3] = self.T
        self.world_view_transform = self.world_view_transform.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device="cuda")
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device="cuda")
        )

        # self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        # self.camera_center = self.world_view_transform.inverse()[:3, 3]
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def projection_matrix(self):
        return getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[:3, 3]
    
    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


    @property
    def Fx(self):
        return fov2focal(self.FoVx, self.image_width)
    
    @property
    def Fy(self):
        return fov2focal(self.FoVy, self.image_height)
    
    @property
    def Cx(self):
        return 0.5 * self.image_width
    
    @property
    def Cy(self):
        return 0.5 * self.image_height
    

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def to_c2w(self):
        return self.world_view_transform.inverse().transpose(0, 1)
    
    def from_c2w(self, c2w: torch.Tensor):
        self.world_view_transform = c2w.inverse().transpose(0, 1)
        

    def resize(self, width, height):
        if width>height:
            self.FoVy = self.FoVx * height / width
        else:
            self.FoVx = self.FoVy * width / height

        # self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        self.image_width = width
        self.image_height = height

    def update_pos(self, move_forward, move_right, move_up, rotate_up, rotate_right, rotate_clockwise):
        
        #这里面的逻辑是先移动，后旋转
        c2w = self.to_c2w()
        R = c2w[:3, :3]

        # 3dgs里面z轴朝前，y轴朝下，x轴朝右
        forward_dir = R@torch.Tensor([0, 0, 1])
        right_dir = R@torch.Tensor([1, 0, 0])
        up_dir = R@torch.Tensor([0, -1, 0])

        c2w[:3, 3]+=move_forward*forward_dir+move_right*right_dir+move_up*up_dir

        #依次解决rotate_up,rotate_right,rotate_clockwise
        # 将角度转换为弧度
        rotate_up_rad = torch.deg2rad(torch.tensor(rotate_up))
        rotate_right_rad = torch.deg2rad(torch.tensor(rotate_right))
        rotate_clockwise_rad = torch.deg2rad(torch.tensor(rotate_clockwise))
        
        # 创建绕局部轴的旋转矩阵
        # 1. 先绕右轴旋转(pitch/up-down)
        if rotate_up != 0:
            cos_up = torch.cos(rotate_up_rad)
            sin_up = torch.sin(rotate_up_rad)
            # 绕右轴的旋转矩阵
            R_pitch = torch.tensor([
                [1, 0, 0],
                [0, cos_up, -sin_up],
                [0, sin_up, cos_up]
            ], dtype=R.dtype, device=R.device)
            R = R @ R_pitch
        
        # 2. 再绕上轴旋转(yaw/left-right)
        if rotate_right != 0:
            cos_right = torch.cos(rotate_right_rad)
            sin_right = torch.sin(rotate_right_rad)
            # 绕上轴的旋转矩阵
            R_yaw = torch.tensor([
                [cos_right, 0, sin_right],
                [0, 1, 0],
                [-sin_right, 0, cos_right]
            ], dtype=R.dtype, device=R.device)
            R = R @ R_yaw
        
        # 3. 最后绕前轴旋转(roll/clockwise)
        if rotate_clockwise != 0:
            cos_roll = torch.cos(rotate_clockwise_rad)
            sin_roll = torch.sin(rotate_clockwise_rad)
            # 绕前轴的旋转矩阵
            R_roll = torch.tensor([
                [cos_roll, -sin_roll, 0],
                [sin_roll, cos_roll, 0],
                [0, 0, 1]
            ], dtype=R.dtype, device=R.device)
            R = R @ R_roll
        
        # 更新c2w矩阵的旋转部分
        c2w[:3, :3] = R

        # 更新transform_matrix
        self.from_c2w(c2w)
        
    def from_camera(self, cam):
        self.image_width = cam.image_width
        self.image_height = cam.image_height
        self.FoVy = cam.FoVy
        self.FoVx = cam.FoVx
        self.znear = cam.znear
        self.zfar = cam.zfar
        self.world_view_transform = cam.world_view_transform

    

class GaussianViewer(QOpenGLWidget):
    '''
    用来管理一个场景的点云和一个场景的相机，并且能够输出渲染
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.move_speed = 0.1
        self.rotate_speed = 0.02
        self.camera_movement = torch.Tensor([0, 0, 0, 0, 0, 0]).cuda()


        self.scaling_modifier = 1000


        self.render_cam = RenderCam(FoVx=1, FoVy=1,
                             image_width = 512, image_height = 512)

        self.image = None

        self.need_update_image = True

        # 设置定时器用于连续移动
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.handle_continuous_movement)
        self.timer.start(16)  # 约60fps
        # 键盘按键状态
        self.key_states = {}

        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.repaint)
        self.refresh_timer.start(16)

    @property
    def render_width(self):
        return self.render_cam.image_width
    
    @property
    def render_height(self):
        return self.render_cam.image_height

    def get_resolution(self):
        return [self.render_width, self.render_height]
    
    def updateImage(self, image):
        self.image = image
        self.update()
    
    def update_camera(self):
        trans = self.camera_movement[:3]
        rot = self.camera_movement[3:]
        c2w = self.render_cam.to_c2w()
        R = c2w[:3, :3]
        C = c2w[:3, 3] +torch.Tensor(trans).cuda() @ R.T * self.move_speed
        c2w[:3, 3] = C

        rot = rot * self.rotate_speed
        Rz = torch.Tensor([
            [torch.cos(rot[2]), -torch.sin(rot[2]), 0],
            [torch.sin(rot[2]),  torch.cos(rot[2]), 0],
            [0, 0, 1]
        ]).cuda()
        Ry = torch.Tensor([
            [torch.cos(rot[1]), 0, -torch.sin(rot[1])],
            [0, 1, 0],
            [torch.sin(rot[1]), 0,  torch.cos(rot[1])]
        ]).cuda()
        Rx = torch.Tensor([
            [1, 0, 0],
            [0, torch.cos(rot[0]), -torch.sin(rot[0])],
            [0, torch.sin(rot[0]),  torch.cos(rot[0])]
        ]).cuda()
        R = R @ Rz @ Ry @ Rx
        c2w[:3, :3] = R
        self.render_cam.from_c2w(c2w)
    

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex_id = glGenTextures(1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        if self.image is None:
            return

        h, w, _ = self.image.shape

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            w, h, 0,
            GL_RGB, GL_UNSIGNED_BYTE,
            self.image
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()

    def update_image(self, image: np.ndarray):
        self.image = image
        self.update()
        # self.repaint()
        self.need_update_image = False


    def keyPressEvent(self, event):
        """处理按键按下事件"""
        key = event.key()
        
        # 记录按键状态
        self.key_states[key] = True
        
        # 处理特殊按键
        # if key == QtCore.Qt.Key_Escape:
        #     self.close()
        
        event.accept()

    def keyReleaseEvent(self, event):
        """处理按键释放事件"""
        key = event.key()
        self.key_states[key] = False
        event.accept()

    def handle_continuous_movement(self):
        """处理连续移动（定时器调用）"""
        # if not any(self.key_states.values()):
        #     return
        
        # 重置相机移动向量
        self.camera_movement*=0.8
        
        # 处理移动
        if self.key_states.get(QtCore.Qt.Key_W):
            self.camera_movement[2] = 1  # 前进
        if self.key_states.get(QtCore.Qt.Key_S):
            self.camera_movement[2] = -1  # 后退
        if self.key_states.get(QtCore.Qt.Key_A):
            self.camera_movement[0] = -1  # 左移
        if self.key_states.get(QtCore.Qt.Key_D):
            self.camera_movement[0] = 1  # 右移
        if self.key_states.get(QtCore.Qt.Key_Q):
            self.camera_movement[1] = 1  # 上升
        if self.key_states.get(QtCore.Qt.Key_E):
            self.camera_movement[1] = -1  # 下降
            
        # 处理旋转
        if self.key_states.get(QtCore.Qt.Key_I):
            self.camera_movement[3] = 1  # 抬头
        if self.key_states.get(QtCore.Qt.Key_K):
            self.camera_movement[3] = -1  # 低头
        if self.key_states.get(QtCore.Qt.Key_J):
            self.camera_movement[4] = 1  # 左转
        if self.key_states.get(QtCore.Qt.Key_L):
            self.camera_movement[4] = -1  # 右转
        if self.key_states.get(QtCore.Qt.Key_O):
            self.camera_movement[5] = 1  # 顺时针滚转
        if self.key_states.get(QtCore.Qt.Key_U):
            self.camera_movement[5] = -1  # 逆时针滚转

        # print(self.camera_movement)
        
        # 如果有任何移动，更新相机
        if self.camera_movement.abs().sum() > 0:
            self.update_camera()
            self.need_update_image = True
        # else:
        #     self.need_update_image = False
    


# class RenderView():
#     def __init__(self, model_path, render_width, render_height):
#         glumpy_app.use("qt5")
#         self.view = glumpy_app.Window(color=(1, 1, 1, 1))
#         self.screen = None
#         import pycuda.gl.autoinit
#         self.view._native_window
#         self.view.set_handler("on_init", self.on_init)
#         self.view.set_handler("on_draw", self.on_draw)
#         self.view.set_handler("on_resize", self.on_resize)

#         if model_path == "":
#             model_path="/home/xjc/code/PGSR/output/rubble_block1/point_cloud/iteration_30000/point_cloud.ply"

#         self.gaussian = GaussianRenderer(render_width, render_height, sh = 3, model_path=model_path)
#         self.gaussian_list = [self.gaussian]
#         self.gaussian_idx = 0
        
#         self.w, self.h = render_width,render_height
#         self.view._native_window.setMinimumSize(self.w, self.h)
#         # self.state = torch.zeros([1, 3, self.h, self.w], dtype=torch.float32, device='cuda')
#         self.create_shared_texture(self.w,self.h,4)

#         vertex_ = """
#         uniform float scale;
#         attribute vec2 position;
#         attribute vec2 texcoord;
#         varying vec2 v_texcoord;
#         void main()
#         {
#             v_texcoord = texcoord;
#             gl_Position = vec4(scale*position, 0.0, 1.0);
#         } """
#         fragment_ = """
#         uniform sampler2D tex;
#         varying vec2 v_texcoord;
#         void main()
#         {
#             gl_FragColor = texture2D(tex, v_texcoord);
#         } """
#         # Build the program and corresponding buffers (with 4 vertices)
#         self.screen = gloo.Program(vertex_, fragment_, count=4)
#         # Upload data into GPU
#         self.screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
#         self.screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
#         self.screen['scale'] = 1.0
#         self.screen['tex'] = self.tex



        
#         @self.view.event
#         def on_mouse_press(x, y, mouse_button):
#             pass
#             # show_w, show_h = self.view._native_window.size().width(), self.view._native_window.size().height()
#             # gaus_w, gaus_h = self.gaussian.get_resolution()
#             # if (show_h / show_w < gaus_h / gaus_w):
#             #     w = show_w
#             #     h = w / gaus_w * gaus_h
#             # else:
#             #     h = show_h
#             #     w = h / gaus_h * gaus_w

#             # # print(f'Mouse pressed in view space {x, y} {self.gaussian.prcp}')
#             # # x -= (self.gaussian.prcp[0] - 0.5) * w
#             # y += 2 * (self.gaussian.prcp[1] - 0.5) * h
#             # # print(f'Mouse pressed in scale space {x, y} {w, h}')
#             # self.gaussian.select_2dpos[0] = x / w * gaus_w
#             # self.gaussian.select_2dpos[1] = y / h * gaus_h
#             # # print(f'Mouse pressed in render space {self.gaussian.select_2dpos} {w, h} {gaus_w, gaus_h}')

#         @self.view.event
#         def on_mouse_release(x, y, mouse_button):
#             pass
#             # if self.gaussian.select_mode:
#             #     select_3dpos = self.gaussian.get_selected_3dpos_v2()
#             #     print(f'Selected 3d pos: {select_3dpos}')
#             #     self.gaussian.select_mask.zero_()
#             #     select_3dpos = torch.tensor(select_3dpos, device = 'cuda')
#             #     distances = torch.norm(self.gaussian.keyPoint - select_3dpos, dim=1).unsqueeze(1)
#             #     index = torch.nonzero(distances < 1.0, as_tuple=False).squeeze().tolist()
#             #     if len(index) != 0:
#             #         self.image_view = ImageView()
#             #         self.image_view.set_image(self.gaussian.img_paths[index[0]])
#             #         self.image_view.show()
#             # self.gaussian.select_2dpos[0] = -1
#             # self.gaussian.select_2dpos[1] = -1
#             # # print(f'Mouse released at ({x}, {y}, {mouse_button}) in render view')

        
#         @self.view.event
#         def on_key_press(k, mod):
#             # print(Key.A, 0x061, k) # weird bugs in keymap, letters should subtract 0x20
#             camera_movement = self.gaussian.camera_movement
#             if k == Key.D - 0x20:
#                 camera_movement[0] = 1
#             if k == Key.A - 0x20:
#                 camera_movement[0] = -1
#             if k == Key.Q - 0x20:
#                 camera_movement[1] = 1
#             if k == Key.E - 0x20:
#                 camera_movement[1] = -1
#             if k == Key.W - 0x20:
#                 camera_movement[2] = 1
#             if k == Key.S - 0x20:
#                 camera_movement[2] = -1        
#             if k == Key.I - 0x20:
#                 camera_movement[3] = 1
#             if k == Key.K - 0x20:
#                 camera_movement[3] = -1
#             if k == Key.J - 0x20:
#                 camera_movement[4] = 1
#             if k == Key.L - 0x20:
#                 camera_movement[4] = -1
#             if k == Key.O - 0x20:
#                 camera_movement[5] = 1
#             if k == Key.U - 0x20:
#                 camera_movement[5] = -1
                

#         @self.view.event
#         def on_key_release(k, mod):
#             camera_movement = self.gaussian.camera_movement
#             if k == Key.D - 0x20:
#                 camera_movement[0] = 0
#             if k == Key.A - 0x20:
#                 camera_movement[0] = 0
#             if k == Key.Q - 0x20:
#                 camera_movement[1] = 0
#             if k == Key.E - 0x20:
#                 camera_movement[1] = 0
#             if k == Key.W - 0x20:
#                 camera_movement[2] = 0
#             if k == Key.S - 0x20:
#                 camera_movement[2] = 0        
#             if k == Key.I - 0x20:
#                 camera_movement[3] = 0
#             if k == Key.K - 0x20:
#                 camera_movement[3] = 0
#             if k == Key.J - 0x20:
#                 camera_movement[4] = 0
#             if k == Key.L - 0x20:
#                 camera_movement[4] = 0
#             if k == Key.O - 0x20:
#                 camera_movement[5] = 0
#             if k == Key.U - 0x20:
#                 camera_movement[5] = 0


#     @contextmanager
#     def cuda_activate(self, img):
#         """Context manager simplifying use of pycuda.gl.RegisteredImage"""
#         mapping = img.map()
#         yield mapping.array(0,0)
#         mapping.unmap()

#     def create_shared_texture(self, w, h, c=4,
#             map_flags=graphics_map_flags.WRITE_DISCARD,
#             dtype=np.uint8):
#         """Create and return a Texture2D with gloo and pycuda views."""
#         self.tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
#         self.tex.activate() # force gloo to create on GPU
#         self.tex.deactivate()
#         self.cuda_buffer = pycuda.gl.RegisteredImage(
#             int(self.tex.handle), self.tex.target, map_flags)
        
#         if self.screen!=None:
#             self.screen['tex'] = self.tex

#     def on_init(self):
#         gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#         gl.glPolygonOffset(1, 1)
#         gl.glEnable(gl.GL_LINE_SMOOTH)
#         gl.glLineWidth(2.5)

#     def on_draw(self, dt):

#         self.gaussian.update_camera()

#         h,w = self.tex.shape[:2]
#         self.rgb, self.depth, self.normal = self.gaussian.render()
        
#         img = torch.clamp(self.rgb, 0, 1)
#         tensor = img.squeeze().permute([1, 2, 0]).flip(0).data # put in texture order
#         tensor = torch.cat((tensor, tensor[:,:,:1]),2) # add the alpha channel
#         tensor[:,:,3] = 1 # set alpha
#         tensor = (255*tensor).byte().contiguous() # convert to ByteTensor
#         assert self.tex.nbytes == tensor.numel()*tensor.element_size()
#         with self.cuda_activate(self.cuda_buffer) as ary:
#             cpy = pycuda.driver.Memcpy2D()
#             cpy.set_src_device(tensor.data_ptr())
#             cpy.set_dst_array(ary)
#             cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = self.tex.nbytes//h
#             cpy.height = h
#             cpy(aligned=False)
#             torch.cuda.synchronize()
#         # window.clear()
#         self.screen.draw(gl.GL_TRIANGLE_STRIP)
       

#     def on_resize(self, width, height):
#         show_w, show_h = self.view._native_window.size().width(), self.view._native_window.size().height()
#         self.gaussian.render_cam.resize(show_w, show_h)
#         self.create_shared_texture(show_w, show_h)
        
        

#     def on_sclaing_slider(self, value):
#         self.gaussian.scaling_modifier = value
#     def on_use_free_mode(self, is_checked):
#         if is_checked:
#             self.gaussian.render_cam = copy.copy(self.gaussian.play_cam)
#             self.gaussian.play_mode = False
#     def on_use_play_mode(self, is_checked):
#         if is_checked:
#             self.gaussian.play_mode = True
#             self.gaussian.play_pause = False
#             self.gaussian.select_mode = False

#     def on_move_speed(self, value):
#         self.gaussian.move_speed = value * 0.1
#         self.view._native_window.setFocus()

#     def on_rotate_speed(self, value):
#         self.gaussian.rotate_speed = value * 0.02
#         self.view._native_window.setFocus()

#     def addGaussian(self, model_path, render_width, render_height):
#         self.gaussian_list.append(GaussianRenderer(render_width, render_height, model_path))
#         # if self.gaussian_idx == -1:
#         #     self.gaussian_idx = 0
#         #     self.gaussian = self.gaussian_list[self.gaussian_idx]
#         return len(self.gaussian_list) - 1

#     def deleteGaussian(self, idx):
#         # print(f'Delete: {idx}/{len(self.gaussian_list)}', end=' ')
#         del self.gaussian_list[idx]
#         torch.cuda.empty_cache()
#         self.gaussian_idx = min(1, len(self.gaussian_list) - 1)
#         self.view._native_window.setFocus()

#         # print(f'-> {self.gaussian_idx}/{len(self.gaussian_list)}')

#     def changeGaussianIdx(self, idx, scaling_modifier, play_mode, move_speed, rotate_speed, label_key_count):
#         # print(f'Change: {self.gaussian_idx}/{len(self.gaussian_list)} -> {idx}/{len(self.gaussian_list)}')
#         self.gaussian_idx = idx
#         # print(len(self.gaussian_list), self.gaussian_idx)
#         self.gaussian = self.gaussian_list[self.gaussian_idx]
#         self.gaussian.scaling_modifier = scaling_modifier
#         self.gaussian.play_mode = play_mode
#         self.gaussian.move_speed = move_speed * 0.1
#         self.gaussian.rotate_speed = rotate_speed * 0.02

#         label_key_count.setText(f'关键帧数量：{len(self.gaussian.key_views)}')
#         self.view._native_window.setFocus()

