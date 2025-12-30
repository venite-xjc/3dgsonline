import sys
from contextlib import contextmanager
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.Qt import QPixmap, QPoint, Qt, QPainter, QIcon
import numpy as np
import time
from glumpy import app as glumpy_app, gl, gloo
from glumpy.app.window import key as Key
import torch
CUDA_ENABLE_GL = True
from pycuda.gl import graphics_map_flags
import pycuda.driver
sys.path.append('/home/xjc/code/LargePGSR/PGSR')
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, focal2fov, fov2focal
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from torchvision.io import read_image
import random
import copy
import os
import json


from scene.cameras import Camera


@torch.no_grad()
class RenderCam():
    def __init__(self, image_width=1600, image_height=1200, FoVx = 1, FoVy=0.75):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy


        self.R = -torch.eye(3).cuda()
        self.T = torch.zeros(3).cuda()

        self.world_view_transform = torch.eye(4).cuda()
        self.world_view_transform[:3, :3] = self.R
        self.world_view_transform[:3, 3] = self.T

        self.zfar = 100.0
        self.znear = 0.01

        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        # self.camera_center = self.world_view_transform.inverse()[:3, 3]
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

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
        

        # print(self.world_view_transform)

    def resize(self, width, height):
        if width>height:
            self.FoVy = self.FoVx * height / width
        else:
            self.FoVx = self.FoVy * width / height

        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        self.image_width = width
        self.image_height = height

    def update_pos(self, move_forward, move_right, move_up, rotate_up, rotate_right, rotate_clockwise):
        
        #这里面的逻辑是先移动，后旋转
        c2w = self.world_view_transform.inverse()
        R = c2w[:3, :3]

        # 3dgs里面z轴朝前，y轴朝下，x轴朝右
        forward_dir = R@torch.Tensor([0, 0, 1])
        right_dir = R@torch.Tensor([1, 0, 0])
        up_dir = R@torch.Tensor([0, -1, 0])

        c2w[:3, 3]+=move_forward*forward_dir+move_right*right_dir+move_up*up_dir

        #先绕x轴旋转再绕y轴旋转再绕z轴旋转
        


class GaussianRenderer:
    '''
    用来管理一个场景的点云和一个场景的相机，并且能够输出渲染
    '''
    def __init__(self, render_width, render_height, sh = 3, model_path=None):

        # 只是用于后面的render函数，并没有什么特殊意义
        parser = ArgumentParser(description="Testing script parameters")
        self.pipeline = PipelineParams(parser)

        # 加载点云
        with torch.no_grad():
            self.gaussians = GaussianModel(sh)
            self.gaussians.load_ply(model_path)
            self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")


        self.move_speed = 0.1
        self.rotate_speed = 0.02
        self.camera_movement = torch.Tensor([0, 0, 0, 0, 0, 0]).cuda()


        self.scaling_modifier = 1000


        self.render_cam = RenderCam(FoVx=1, FoVy=render_height/render_width,
                             image_width = render_width, image_height = render_height)

    @property
    def render_width(self):
        return self.render_cam.image_width
    
    @property
    def render_height(self):
        return self.render_cam.image_height

    def get_resolution(self):
        return [self.render_width, self.render_height]
    
    def render(self):
        with torch.no_grad():
            rendering = render(self.render_cam, self.gaussians, self.pipeline, self.background, 
                               1, override_color=None, app_model=None, 
                               return_plane=True, return_depth_normal=True)
            self.render_rgb, self.render_depth, self.render_normal = rendering['render'], rendering["plane_depth"], rendering["rendered_normal"]
            self.render_normal = (self.render_normal+1)/2
        return self.render_rgb, self.render_depth, self.render_normal
    
    
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
    



class RenderView():
    def __init__(self, model_path, render_width, render_height):
        glumpy_app.use("qt5")
        self.view = glumpy_app.Window(color=(1, 1, 1, 1))
        self.screen = None
        import pycuda.gl.autoinit
        self.view._native_window
        self.view.set_handler("on_init", self.on_init)
        self.view.set_handler("on_draw", self.on_draw)
        self.view.set_handler("on_resize", self.on_resize)

        if model_path == "":
            model_path="/home/xjc/code/PGSR/output/rubble_block1/point_cloud/iteration_30000/point_cloud.ply"

        self.gaussian = GaussianRenderer(render_width, render_height, sh = 3, model_path=model_path)
        self.gaussian_list = [self.gaussian]
        self.gaussian_idx = 0
        
        self.w, self.h = render_width,render_height
        self.view._native_window.setMinimumSize(self.w, self.h)
        # self.state = torch.zeros([1, 3, self.h, self.w], dtype=torch.float32, device='cuda')
        self.create_shared_texture(self.w,self.h,4)

        vertex_ = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        fragment_ = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """
        # Build the program and corresponding buffers (with 4 vertices)
        self.screen = gloo.Program(vertex_, fragment_, count=4)
        # Upload data into GPU
        self.screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        self.screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = self.tex



        
        @self.view.event
        def on_mouse_press(x, y, mouse_button):
            pass
            # show_w, show_h = self.view._native_window.size().width(), self.view._native_window.size().height()
            # gaus_w, gaus_h = self.gaussian.get_resolution()
            # if (show_h / show_w < gaus_h / gaus_w):
            #     w = show_w
            #     h = w / gaus_w * gaus_h
            # else:
            #     h = show_h
            #     w = h / gaus_h * gaus_w

            # # print(f'Mouse pressed in view space {x, y} {self.gaussian.prcp}')
            # # x -= (self.gaussian.prcp[0] - 0.5) * w
            # y += 2 * (self.gaussian.prcp[1] - 0.5) * h
            # # print(f'Mouse pressed in scale space {x, y} {w, h}')
            # self.gaussian.select_2dpos[0] = x / w * gaus_w
            # self.gaussian.select_2dpos[1] = y / h * gaus_h
            # # print(f'Mouse pressed in render space {self.gaussian.select_2dpos} {w, h} {gaus_w, gaus_h}')

        @self.view.event
        def on_mouse_release(x, y, mouse_button):
            pass
            # if self.gaussian.select_mode:
            #     select_3dpos = self.gaussian.get_selected_3dpos_v2()
            #     print(f'Selected 3d pos: {select_3dpos}')
            #     self.gaussian.select_mask.zero_()
            #     select_3dpos = torch.tensor(select_3dpos, device = 'cuda')
            #     distances = torch.norm(self.gaussian.keyPoint - select_3dpos, dim=1).unsqueeze(1)
            #     index = torch.nonzero(distances < 1.0, as_tuple=False).squeeze().tolist()
            #     if len(index) != 0:
            #         self.image_view = ImageView()
            #         self.image_view.set_image(self.gaussian.img_paths[index[0]])
            #         self.image_view.show()
            # self.gaussian.select_2dpos[0] = -1
            # self.gaussian.select_2dpos[1] = -1
            # # print(f'Mouse released at ({x}, {y}, {mouse_button}) in render view')

        
        @self.view.event
        def on_key_press(k, mod):
            # print(Key.A, 0x061, k) # weird bugs in keymap, letters should subtract 0x20
            camera_movement = self.gaussian.camera_movement
            if k == Key.D - 0x20:
                camera_movement[0] = 1
            if k == Key.A - 0x20:
                camera_movement[0] = -1
            if k == Key.Q - 0x20:
                camera_movement[1] = 1
            if k == Key.E - 0x20:
                camera_movement[1] = -1
            if k == Key.W - 0x20:
                camera_movement[2] = 1
            if k == Key.S - 0x20:
                camera_movement[2] = -1        
            if k == Key.I - 0x20:
                camera_movement[3] = 1
            if k == Key.K - 0x20:
                camera_movement[3] = -1
            if k == Key.J - 0x20:
                camera_movement[4] = 1
            if k == Key.L - 0x20:
                camera_movement[4] = -1
            if k == Key.O - 0x20:
                camera_movement[5] = 1
            if k == Key.U - 0x20:
                camera_movement[5] = -1
                

        @self.view.event
        def on_key_release(k, mod):
            camera_movement = self.gaussian.camera_movement
            if k == Key.D - 0x20:
                camera_movement[0] = 0
            if k == Key.A - 0x20:
                camera_movement[0] = 0
            if k == Key.Q - 0x20:
                camera_movement[1] = 0
            if k == Key.E - 0x20:
                camera_movement[1] = 0
            if k == Key.W - 0x20:
                camera_movement[2] = 0
            if k == Key.S - 0x20:
                camera_movement[2] = 0        
            if k == Key.I - 0x20:
                camera_movement[3] = 0
            if k == Key.K - 0x20:
                camera_movement[3] = 0
            if k == Key.J - 0x20:
                camera_movement[4] = 0
            if k == Key.L - 0x20:
                camera_movement[4] = 0
            if k == Key.O - 0x20:
                camera_movement[5] = 0
            if k == Key.U - 0x20:
                camera_movement[5] = 0


    @contextmanager
    def cuda_activate(self, img):
        """Context manager simplifying use of pycuda.gl.RegisteredImage"""
        mapping = img.map()
        yield mapping.array(0,0)
        mapping.unmap()

    def create_shared_texture(self, w, h, c=4,
            map_flags=graphics_map_flags.WRITE_DISCARD,
            dtype=np.uint8):
        """Create and return a Texture2D with gloo and pycuda views."""
        self.tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
        self.tex.activate() # force gloo to create on GPU
        self.tex.deactivate()
        self.cuda_buffer = pycuda.gl.RegisteredImage(
            int(self.tex.handle), self.tex.target, map_flags)
        
        if self.screen!=None:
            self.screen['tex'] = self.tex

    def on_init(self):
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPolygonOffset(1, 1)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(2.5)

    def on_draw(self, dt):

        self.gaussian.update_camera()

        h,w = self.tex.shape[:2]
        self.rgb, self.depth, self.normal = self.gaussian.render()
        
        img = torch.clamp(self.rgb, 0, 1)
        tensor = img.squeeze().permute([1, 2, 0]).flip(0).data # put in texture order
        tensor = torch.cat((tensor, tensor[:,:,:1]),2) # add the alpha channel
        tensor[:,:,3] = 1 # set alpha
        tensor = (255*tensor).byte().contiguous() # convert to ByteTensor
        assert self.tex.nbytes == tensor.numel()*tensor.element_size()
        with self.cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = self.tex.nbytes//h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()
        # window.clear()
        self.screen.draw(gl.GL_TRIANGLE_STRIP)
       

    def on_resize(self, width, height):
        show_w, show_h = self.view._native_window.size().width(), self.view._native_window.size().height()
        self.gaussian.render_cam.resize(show_w, show_h)
        self.create_shared_texture(show_w, show_h)
        
        

    def on_sclaing_slider(self, value):
        self.gaussian.scaling_modifier = value
    def on_use_free_mode(self, is_checked):
        if is_checked:
            self.gaussian.render_cam = copy.copy(self.gaussian.play_cam)
            self.gaussian.play_mode = False
    def on_use_play_mode(self, is_checked):
        if is_checked:
            self.gaussian.play_mode = True
            self.gaussian.play_pause = False
            self.gaussian.select_mode = False

    def on_move_speed(self, value):
        self.gaussian.move_speed = value * 0.1
        self.view._native_window.setFocus()

    def on_rotate_speed(self, value):
        self.gaussian.rotate_speed = value * 0.02
        self.view._native_window.setFocus()

    def addGaussian(self, model_path, render_width, render_height):
        self.gaussian_list.append(GaussianRenderer(render_width, render_height, model_path))
        # if self.gaussian_idx == -1:
        #     self.gaussian_idx = 0
        #     self.gaussian = self.gaussian_list[self.gaussian_idx]
        return len(self.gaussian_list) - 1

    def deleteGaussian(self, idx):
        # print(f'Delete: {idx}/{len(self.gaussian_list)}', end=' ')
        del self.gaussian_list[idx]
        torch.cuda.empty_cache()
        self.gaussian_idx = min(1, len(self.gaussian_list) - 1)
        self.view._native_window.setFocus()

        # print(f'-> {self.gaussian_idx}/{len(self.gaussian_list)}')

    def changeGaussianIdx(self, idx, scaling_modifier, play_mode, move_speed, rotate_speed, label_key_count):
        # print(f'Change: {self.gaussian_idx}/{len(self.gaussian_list)} -> {idx}/{len(self.gaussian_list)}')
        self.gaussian_idx = idx
        # print(len(self.gaussian_list), self.gaussian_idx)
        self.gaussian = self.gaussian_list[self.gaussian_idx]
        self.gaussian.scaling_modifier = scaling_modifier
        self.gaussian.play_mode = play_mode
        self.gaussian.move_speed = move_speed * 0.1
        self.gaussian.rotate_speed = rotate_speed * 0.02

        label_key_count.setText(f'关键帧数量：{len(self.gaussian.key_views)}')
        self.view._native_window.setFocus()

class HelpView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("操作帮助")
        self.setFixedSize(800, 900)
        self.setStyleSheet("background-color: rgb(70, 70, 70);")
        self.keys = {
            'W': QtWidgets.QLabel('W', self),
            'A': QtWidgets.QLabel('A', self),
            'S': QtWidgets.QLabel('S', self),
            'D': QtWidgets.QLabel('D', self),
            'Q': QtWidgets.QLabel('Q', self),
            'E': QtWidgets.QLabel('E', self),
            'U': QtWidgets.QLabel('U', self),
            'I': QtWidgets.QLabel('I', self),
            'O': QtWidgets.QLabel('O', self),
            'J': QtWidgets.QLabel('J', self),
            'K': QtWidgets.QLabel('K', self),
            'L': QtWidgets.QLabel('L', self),
            'Space': QtWidgets.QLabel('空格', self),
            'Space1': QtWidgets.QLabel('空格', self),
        }
        for i in self.keys.values():
            i.setFixedSize(80, 80)
            i.setStyleSheet("border: 3px solid rgb(0, 200, 200); border-radius: 10px")
            i.setAlignment(Qt.AlignCenter)
        self.keys['Space'].setFixedSize(3 * 80 - 10, 80)
        self.keys['Space1'].setFixedSize(3 * 80 - 10, 80)
            
        self.labels = {
            'free_mode': QtWidgets.QLabel('自由视角模式:', self),
            'play_mode': QtWidgets.QLabel('轨迹播放模式:', self),
            'move': QtWidgets.QLabel('移动视角', self),
            'rotate': QtWidgets.QLabel('旋转视角', self),
            'pause': QtWidgets.QLabel('暂停轨迹播放 / 继续轨迹播放', self),
            'select': QtWidgets.QLabel('开启点击选择 / 关闭点击选择 ', self),
            'show_image': QtWidgets.QLabel('点击显示高清图像', self),

        }
        self.labels['free_mode'].setStyleSheet("color: rgba( 255, 255, 255, 100% ); font-weight: bold;")
        self.labels['play_mode'].setStyleSheet("color: rgba( 255, 255, 255, 100% ); font-weight: bold;")

        
        self.total_layout = QtWidgets.QVBoxLayout(self)


        self.free_mode = QtWidgets.QWidget()
        # self.free_mode.setStyleSheet("background-color: rgba(0, 200, 200, 50);")
        self.vlayout0 = QtWidgets.QVBoxLayout(self.free_mode)
        
        self.keycomb0 = QtWidgets.QWidget(self.free_mode)
        self.keycomb0.setFixedSize(3 * 80, 2 * 80)
        self.keycomb0_l = QtWidgets.QVBoxLayout(self.keycomb0)
        self.keycomb0_l0 = QtWidgets.QHBoxLayout()
        self.keycomb0_l1 = QtWidgets.QHBoxLayout()
        # self.keycomb0_l.setContentsMargins(0, 0, 0, 0)
        # self.keycomb0_l0.setContentsMargins(0, 0, 0, 0)
        # self.keycomb0_l1.setContentsMargins(0, 0, 0, 0)
        self.keycomb0_l.addLayout(self.keycomb0_l0)
        self.keycomb0_l.addLayout(self.keycomb0_l1)
        self.keycomb0_l0.addWidget(self.keys['Q'])
        self.keycomb0_l0.addWidget(self.keys['W'])
        self.keycomb0_l0.addWidget(self.keys['E'])
        self.keycomb0_l1.addWidget(self.keys['A'])
        self.keycomb0_l1.addWidget(self.keys['S'])
        self.keycomb0_l1.addWidget(self.keys['D'])

        self.hlayout0 = QtWidgets.QHBoxLayout()
        self.hlayout0.addWidget(self.keycomb0)
        self.hlayout0.addWidget(self.labels['move'])


        self.keycomb1 = QtWidgets.QWidget(self.free_mode)
        self.keycomb1.setFixedSize(3 * 80, 2 * 80)
        self.keycomb1_l = QtWidgets.QVBoxLayout(self.keycomb1)
        self.keycomb1_l0 = QtWidgets.QHBoxLayout()
        self.keycomb1_l1 = QtWidgets.QHBoxLayout()
        # self.keycomb1_l.setContentsMargins(0, 0, 0, 0)
        # self.keycomb1_l0.setContentsMargins(0, 0, 0, 0)
        # self.keycomb1_l1.setContentsMargins(0, 0, 0, 0)
        self.keycomb1_l.addLayout(self.keycomb1_l0)
        self.keycomb1_l.addLayout(self.keycomb1_l1)
        self.keycomb1_l0.addWidget(self.keys['U'])
        self.keycomb1_l0.addWidget(self.keys['I'])
        self.keycomb1_l0.addWidget(self.keys['O'])
        self.keycomb1_l1.addWidget(self.keys['J'])
        self.keycomb1_l1.addWidget(self.keys['K'])
        self.keycomb1_l1.addWidget(self.keys['L'])

        self.hlayout1 = QtWidgets.QHBoxLayout()
        self.hlayout1.addWidget(self.keycomb1)
        self.hlayout1.addWidget(self.labels['rotate'])

        self.empty3 = QtWidgets.QWidget()
        self.empty3.setFixedHeight(10)

        self.hlayout3 = QtWidgets.QHBoxLayout()
        self.empty0 = QtWidgets.QWidget()
        self.empty0.setFixedWidth(6)
        self.hlayout3.addWidget(self.empty0)
        self.hlayout3.addWidget(self.keys['Space'])
        self.hlayout3.addWidget(self.labels['select'])


        self.hlayout4 = QtWidgets.QHBoxLayout()
        self.empty2 = QtWidgets.QWidget()
        self.empty2.setFixedWidth(6)
        self.hlayout4.addWidget(self.empty2)
        self.image = QPixmap('./images/select.png')
        self.image = self.image.scaled(self.image.width() // 10, self.image.height() // 10)
        self.show_image = QtWidgets.QLabel()
        self.show_image.setFixedSize(3 * 80 - 10, 80)
        self.show_image.setAlignment(QtCore.Qt.AlignCenter)
        self.show_image.setPixmap(self.image)
        self.hlayout4.addWidget(self.show_image)
        self.hlayout4.addWidget(self.labels['show_image'])

        self.vlayout0.addWidget(self.labels['free_mode'])
        # self.labels['free_mode'].setAlignment(QtCore.Qt.AlignCenter)
        self.vlayout0.addLayout(self.hlayout0)
        self.vlayout0.addLayout(self.hlayout1)
        self.vlayout0.addWidget(self.empty3)
        self.vlayout0.addLayout(self.hlayout3)
        self.vlayout0.addLayout(self.hlayout4)


        self.play_mode = QtWidgets.QWidget()
        self.vlayout1 = QtWidgets.QVBoxLayout(self.play_mode)
        self.hlayout2 = QtWidgets.QHBoxLayout()
        self.empty1 = QtWidgets.QWidget()
        self.empty1.setFixedWidth(6)
        self.hlayout2.addWidget(self.empty1)
        self.hlayout2.addWidget(self.keys['Space1'])
        self.hlayout2.addWidget(self.labels['pause'])
        self.vlayout1.addWidget(self.labels['play_mode'])
        self.vlayout1.addLayout(self.hlayout2)



        self.total_layout.addWidget(self.free_mode)
        self.total_layout.addWidget(self.play_mode)

class ImageView(QtWidgets.QWidget):
    def __init__(self):
        super(ImageView, self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0, 0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 1.0
        self.setWindowTitle("图像")

    def set_image(self, img_path):
        self.img = QPixmap(img_path)
        self.scaled_img = self.img
        # self.setBaseSize(self.scaled_img.width(), self.scaled_img.height())
        self.setMinimumSize(self.scaled_img.width(), self.scaled_img.height())

    def paintEvent(self, e):
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(self.point, self.scaled_img)
            painter.end()

    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        angleY = angle.y()
        
        if angleY > 0:
            self.scale *= 1.1
        else:
            self.scale *= 0.9
        self.adjustSize()
        self.update()


    def mouseMoveEvent(self, e):
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos / self.scale
            self.start_pos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()


class Window(QtWidgets.QMainWindow):
    def __init__(self, render_width, render_height, path = ""):
        super().__init__()
        
        
        self.default_model_path = path
        self.glumpy_window = RenderView(self.default_model_path,render_width, render_height)

        self.setMaximumSize(1920 * 2, 1080 * 2)
        
       

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.horizontallayout = QtWidgets.QHBoxLayout(self.centralwidget)

        


        # self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.scale_slider.setMinimum(0)
        # self.scale_slider.setMaximum(100)
        # self.scale_slider.setValue(100)
        # self.scale_slider.valueChanged.connect(self.glumpy_window.on_sclaing_slider)
        # self.hlayout_scale_slider = QtWidgets.QHBoxLayout()
        # self.hlayout_scale_slider.addWidget(self.labels['scaling'])
        # self.hlayout_scale_slider.addWidget(self.scale_slider)
        # self.control_layout_render.addLayout(self.hlayout_scale_slider, stretch=0)
        
        

        # self.move_speed = QtWidgets.QDoubleSpinBox()
        # self.move_speed.setRange(0.1, 10.0)
        # self.move_speed.setSingleStep(0.1)
        # self.move_speed.setValue(1.0)
        # self.move_speed.valueChanged.connect(self.glumpy_window.on_move_speed)
        # self.hlayout_move_speed = QtWidgets.QHBoxLayout()
        # self.hlayout_move_speed.addWidget(self.labels['moveSpeed'])
        # self.hlayout_move_speed.addWidget(self.move_speed, stretch=1)
        # self.control_layout_render.addLayout(self.hlayout_move_speed)

        # self.rotate_speed = QtWidgets.QDoubleSpinBox()
        # self.rotate_speed.setRange(0.1, 10.0)
        # self.rotate_speed.setSingleStep(0.1)
        # self.rotate_speed.setValue(1.0)
        # self.rotate_speed.valueChanged.connect(self.glumpy_window.on_rotate_speed)
        # self.hlayout_rotate_speed = QtWidgets.QHBoxLayout()
        # self.hlayout_rotate_speed.addWidget(self.labels['rotSpeed'])
        # self.hlayout_rotate_speed.addWidget(self.rotate_speed, stretch=1)
        # self.control_layout_render.addLayout(self.hlayout_rotate_speed)




        # self.control_widget_scene = QtWidgets.QWidget()
        # self.control_widget_scene.setObjectName("control_panel_scene")
        # self.control_layout_scene = QtWidgets.QVBoxLayout(self.control_widget_scene)

        # self.hlayout_raido_button = QtWidgets.QVBoxLayout()

        # self.scene_button_group = QtWidgets.QButtonGroup(self.control_widget_scene)
        # self.scene_buttons = []
        
        # self.labels['scene_control'].setAlignment(QtCore.Qt.AlignCenter)
        # self.labels['scene_control'].setObjectName("scene_control")
        # self.hlayout_raido_button.addWidget(self.labels['scene_control'])
        # # self.hlayout_raido_button.addWidget(self.radio_button)
        # self.hlayout_raido_button.addWidget(self.buttons['load'])
        # self.buttons['load'].clicked.connect(lambda state, w=render_width, h=render_height: self.load_model(w, h))

        # self.control_layout_scene.addLayout(self.hlayout_raido_button)

        # self.gpu_progress_bar = QtWidgets.QProgressBar()
        # self.gpu_progress_bar.setMinimum(0)
        # self.gpu_progress_bar.setMaximum(100)
        # self.hlayout_gpu_bar = QtWidgets.QHBoxLayout()
        # self.hlayout_gpu_bar.addWidget(self.labels["gpu_bar"])
        # self.hlayout_gpu_bar.addWidget(self.gpu_progress_bar, stretch=0)
        # self.control_layout_scene.addLayout(self.hlayout_gpu_bar)
        # timer = app.timer = QtCore.QTimer()
        # timer.timeout.connect(self.get_gpu_memory)
        # timer.start(500)


        # self.control_widget = QtWidgets.QWidget()
        # self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        # self.control_layout.addWidget(self.control_widget_render)
        # self.control_layout.addWidget(self.control_widget_scene)
        # self.control_widget.setMinimumWidth(400)
        # self.control_layout.setContentsMargins(0, 0, 0, 0)
        
        # self.empty_space = QtWidgets.QWidget()
        # self.control_layout.addWidget(self.empty_space, stretch=1)

        # self.control_layout.addWidget(self.buttons['help'])
        # self.help_view = HelpView()
        # self.buttons['help'].setMaximumWidth(200)
        # self.buttons['help'].setObjectName('help')
        # self.buttons['help'].clicked.connect(self.show_help)

        # self.horizontallayout.addWidget(self.control_widget, stretch=0)


        self.horizontallayout.addWidget(self.glumpy_window.view._native_window, stretch=1)
        self.glumpy_window.view._native_window.setFocus() # so RenderView recieves keyboard events
        self.glumpy_window.view._native_window.setFocusPolicy(QtCore.Qt.StrongFocus)
        time.sleep(1)




    def load_cams(self):
        cams_path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "keycamera.json", "JSON Files (*.json)")[0]
        if os.path.exists(cams_path) is False:
            print('cams path not found')
            return
        self.glumpy_window.gaussian.load_cams(cams_path)

    def add_cam(self):
        self.glumpy_window.gaussian.add_cam()
        self.labels['key_cams_num'].setText(f'关键帧数量：{len(self.glumpy_window.gaussian.key_views)}')
        self.glumpy_window.view._native_window.setFocus()

    def del_cam(self):
        self.glumpy_window.gaussian.del_cam()
        self.labels['key_cams_num'].setText(f'关键帧数量：{len(self.glumpy_window.gaussian.key_views)}')
        self.glumpy_window.view._native_window.setFocus()
    
    def save_cams(self):
        cams_path = QtWidgets.QFileDialog.getSaveFileName(None, "选取文件", "keycamera.json", "JSON Files (*.json)")[0]
        if cams_path != "":
            self.glumpy_window.gaussian.save_cams(cams_path)

    def show_help(self):
        self.help_view.show()

    def get_gpu_memory(self):
        import subprocess
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'])
            result = str(result, encoding='utf-8')
            self.memory_free, self.memory_total = [int(x) for x in result.strip().split('\n')[0].split(',')]
            self.memory_usage = 100 - self.memory_free / self.memory_total * 100
        except Exception as e:
            self.memory_free = 0
            self.memory_total = 1
            self.memory_usage = 100
        self.gpu_progress_bar.setValue(int(100 - self.memory_free / self.memory_total * 100))
        

    def load_model(self, render_width, render_height):
        model_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")
        if os.path.exists(model_path) is False or os.path.exists(os.path.join(model_path, "cfg_args")) is False:
            print('Model path not found')
            return
        button_text = os.path.split(model_path)[1]
        model_count = self.glumpy_window.addGaussian(model_path, render_width, render_height)
        radio_button = QtWidgets.QRadioButton(button_text)
        radio_button.value = model_count
        delete_button = QtWidgets.QPushButton('    删除    ')

        radio_button.toggled.connect(lambda state, button=radio_button, scaling_modifier=self.scale_slider, mode=self.buttons['play'], move=self.move_speed, rot=self.rotate_speed, key_count=self.labels['key_cams_num']: self.glumpy_window.changeGaussianIdx(button.value, scaling_modifier.value(), mode.isChecked(), move.value(), rot.value(), key_count))
        delete_button.clicked.connect(lambda state, button=radio_button: self.delete_model(button.value))

        empty = QtWidgets.QWidget()
        hwidget = QtWidgets.QWidget()
        hlayout = QtWidgets.QHBoxLayout(hwidget)
        hlayout.addWidget(radio_button)
        hlayout.addWidget(empty, stretch=1)
        hlayout.addWidget(delete_button)
        
        self.hlayout_raido_button.insertWidget(model_count, hwidget)
        self.scene_button_group.addButton(radio_button)
        radio_button.toggle()

    def init_load_model(self, render_width, render_height, model_path):
        # model_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")
        if os.path.exists(model_path) is False or os.path.exists(os.path.join(model_path, "cfg_args")) is False:
            print('Model path not found')
            return
        button_text = os.path.split(model_path)[1]
        model_count = self.glumpy_window.addGaussian(model_path, render_width, render_height)
        radio_button = QtWidgets.QRadioButton(button_text)
        radio_button.value = model_count
        delete_button = QtWidgets.QPushButton('    删除    ')

        radio_button.toggled.connect(lambda state, button=radio_button, scaling_modifier=self.scale_slider, mode=self.buttons['play'], move=self.move_speed, rot=self.rotate_speed, key_count=self.labels['key_cams_num']: self.glumpy_window.changeGaussianIdx(button.value, scaling_modifier.value(), mode.isChecked(), move.value(), rot.value(), key_count))
        delete_button.clicked.connect(lambda state, button=radio_button: self.delete_model(button.value))

        empty = QtWidgets.QWidget()
        hwidget = QtWidgets.QWidget()
        hlayout = QtWidgets.QHBoxLayout(hwidget)
        hlayout.addWidget(radio_button)
        hlayout.addWidget(empty, stretch=1)
        hlayout.addWidget(delete_button)
        
        self.hlayout_raido_button.insertWidget(model_count, hwidget)
        self.scene_button_group.addButton(radio_button)
        radio_button.toggle()

    def delete_model(self, idx):
        self.glumpy_window.deleteGaussian(idx)
        self.hlayout_raido_button.removeWidget(self.hlayout_raido_button.itemAt(idx).widget())
        for i, b in enumerate(self.scene_button_group.buttons()):
            b.value = i + 1
        if self.glumpy_window.gaussian_idx > 0:
            self.scene_button_group.buttons()[self.glumpy_window.gaussian_idx - 1].toggle()
        else:
            self.glumpy_window.changeGaussianIdx(0, self.scale_slider.value(), self.buttons['play'].isChecked(), self.move_speed.value(), self.rotate_speed.value(), self.labels['key_cams_num'])

        

    def showEvent(self, event):
        # print('showevent0', self.glumpy_window.view._native_window.size().width(), self.glumpy_window.view._native_window.size().height())
        super().showEvent(event)
        # print(f'Window is created in size: {self.size().width(), self.size().height()}')
        # self.glumpy_window.view.dispatch_event("on_resize", self.size().width() - 200, self.size().height() - 200)
        # print('showevent1', self.glumpy_window.view._native_window.size().width(), self.glumpy_window.view._native_window.size().height())
        # exit()
        # render_w, render_h = self.glumpy_window.view.get_size()
        # print('showevent1', self.glumpy_window.view._native_window.size().width(), self.glumpy_window.view._native_window.size().height())
        # exit()
        # w = self.size().width() - 200
        # h = w * render_h // render_w
        # # print(width, height)
        # self.glumpy_window.view.dispatch_event("on_resize", 1500, 1000)
        # self.glumpy_window.view.dispatch_event("on_resize", *self.glumpy_window.view.get_size())
        # test_w, test_h = self.glumpy_window.view.get_size()
        # render_w, render_h = self.glumpy_window.gaussian.get_resolution()
        # w = self.glumpy_window.view._native_window.width()
        # h = self.glumpy_window.view._native_window.height()
        # if w / h > render_w / render_h:
        #     h = w * render_h // render_w
        # else:
        #     w = h * render_w // render_h
        # self.glumpy_window.view.dispatch_event("on_resize", w, h)


    def closeEvent(self, event):
        super().closeEvent(event)
        self.glumpy_window.view.close()

    def resizeEvent(self, event):
        # print(f'Window is resized to: ({event.size().width(), event.size().height()})')
        test_w, test_h = self.glumpy_window.view.get_size()
        render_w, render_h = self.glumpy_window.gaussian.get_resolution()
        w = self.glumpy_window.view._native_window.width()
        h = self.glumpy_window.view._native_window.height()
        if w / h > render_w / render_h:
            h = w * render_h // render_w
        else:
            w = h * render_w // render_h
        self.glumpy_window.view.dispatch_event("on_resize", w, h)
        pass

    def mousePressEvent(self, event):
        pos = event.pos()
        # print(f'Mouse pressed at ({pos.x()}, {pos.y()}) in main window')
        
    def load_file(self):
        self.model_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")


    def keyPressEvent(self, event):
        # if event.key() == QtCore.Qt.Key_T:
        #     self.image_view.exec_()
        pass
            
    def keyReleaseEvent(self, event):
        pass

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        # qp.drawPixmap(0, 0, self.bg)
        # print('paint')
        # exit()
            

def apply_stylesheets(qss, app):
    with open(qss, "r") as f:
        _style = f.read()
    app.setStyleSheet(_style)


if __name__ == "__main__":
    
    # glumpy_app.Window(visible=False)
    app = QtWidgets.QApplication([])
    
    # apply_stylesheets('./stylesheets/wendi.qss', app)
    # apply_stylesheets('stylesheets/MaterialDark.qss', app)
    # import qdarkstyle
    # app.setStyleSheet(qdarkstyle.load_stylesheet())
    # import sys
    # ply_path = sys.argv[1]
    ply_path = ""

    render_width, render_height = [1920, 1080]
    window = Window(render_width, render_height, ply_path)
    
    window.show()
    glumpy_app.run()
