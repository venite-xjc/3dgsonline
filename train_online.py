#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from viewer import MainViewerWindow
from PyQt5.QtWidgets import (
    QApplication,)
import sys
# app = QApplication(sys.argv)
# window = MainViewerWindow()
# window.show()
# sys.exit(app.exec_())
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.pose_utils import rt2mat, V, SO3_exp, update_pose
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import UInt32
from nav_msgs.msg import Odometry, Path

from cv_bridge import CvBridge
# import cv2
import numpy as np

import queue
import threading

from receiver import FastLivo2Receiver
import rclpy

from scene.cameras import Camera, MiniCam, OnlineCam
from utils.graphics_utils import focal2fov, fov2focal
from scene.colmap_loader import qvec2rotmat



def add_points_with_voxel_limit(A, B, voxel_size, k, origin=None):
    """
    在体素约束下，从候选点云 B 中选择可以加入的点。
    
    Args:
        A (Tensor): [N, 3], 已有点云
        B (Tensor): [M, 3], 候选点云
        voxel_size (float or Tensor): 体素尺寸，标量或 [3]
        k (int): 每个体素最多允许 A 中有 k 个点，若已满则不加 B 的点
        origin (Tensor, optional): [3], 体素原点，默认为 0
    
    Returns:
        selected_B (Tensor): [L, 3], 从 B 中选出的可加入点
    """
    device = A.device
    if origin is None:
        origin = torch.zeros(1, 3, device=device)
    else:
        origin = origin.to(device)

    voxel_size = torch.as_tensor(voxel_size, device=device).view(-1)
    if voxel_size.numel() == 1:
        voxel_size = voxel_size.expand(3)

    # 步骤1: 计算 A 和 B 的体素坐标（整数索引）
    voxel_coords_A = torch.floor((A - origin) / voxel_size).long()  # [N, 3]
    voxel_coords_B = torch.floor((B - origin) / voxel_size).long()  # [M, 3]

    min_block = torch.min(voxel_coords_A, dim=0)[0]
    max_block = torch.max(voxel_coords_A+1, dim=0)[0]

    min_block2 = torch.min(voxel_coords_B, dim=0)[0]
    max_block2 = torch.max(voxel_coords_B+1, dim=0)[0]


    min_block = torch.min(torch.stack([min_block, min_block2], dim=0), dim=0)[0]
    max_block = torch.max(torch.stack([max_block, max_block2], dim=0), dim=0)[0]

    voxel_coords_A = voxel_coords_A - min_block
    voxel_coords_B = voxel_coords_B - min_block

    unique_voxel_index, counts_A = torch.unique(
        voxel_coords_A, dim=0, return_inverse=False, return_counts=True
    )

    valid_voxel_mask = counts_A > k  # [U], U = 唯一体素数
    full_voxels = unique_voxel_index[valid_voxel_mask]  # [V, 3]

    full_voxel_keys = full_voxels[:, 0] * max_block[1] * max_block[2] + full_voxels[:, 1] * max_block[2] + full_voxels[:, 2]
    B_keys = voxel_coords_B[:, 0] * max_block[1] * max_block[2] + voxel_coords_B[:, 1] * max_block[2] + voxel_coords_B[:, 2]
    

    # 创建一个 mask：B 中哪些点的体素是 valid 的
    # 使用 torch.isin（PyTorch 1.9+）
    mask = torch.isin(B_keys, full_voxel_keys)

    # selected_B = B[mask]
    # return selected_B
    return ~mask


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, empty_init=True)
    # gaussians.training_setup(opt)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    viewpoint_stack = []
    
    rclpy.init()
    receiver = FastLivo2Receiver(queue_max_length=50)

    # 启动 ROS spin 线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(receiver,), daemon=True)
    spin_thread.start()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    last_cam_position = [-10000, -10000, -10000]
    last_cam_rotation = [1, 0, 0]

    # app = QApplication(sys.argv)
    # window = MainViewerWindow()
    # window.show()

    # gaussians.oneupSHdegree()
    # gaussians.oneupSHdegree()

    iteration = 0
    while 1:
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    with torch.no_grad():
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()


        length_to_process = min(receiver.pcd_queue.qsize(), receiver.img_queue.qsize(), receiver.pose_queue.qsize())
        if length_to_process > 0 and iteration % 20 == 0:
            print("Processing %d frames" % length_to_process)
            xyzs_list = []
            colors_list = []
            camera_add_list = []
            for i in range(length_to_process):
                xyzs, colors = receiver.pcd_queue.get()

                xyzs_list.append(xyzs)
                colors_list.append(colors)
                image = receiver.img_queue.get()

                qvec, tvec = receiver.pose_queue.get()


                # print(qvec, tvec, "image id is", str(len(scene.online_cameras)+1))

                R = np.transpose(qvec2rotmat(qvec))
                T = np.array(tvec)
                K = receiver.K
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                image_width = receiver.image_width
                image_height = receiver.image_height
                fov_x = focal2fov(fx, image_width)
                fov_y = focal2fov(fy, image_height)
                
                # W2C = np.zeros((4, 4))
                # W2C[:3, :3] = R.transpose()
                # W2C[:3, 3] = T
                # W2C[3, 3] = 1.0
                # C2W = np.linalg.inv(W2C)

                C2W_R = R
                C2W_T = -R @ T

                lookat = R@np.array([0, 0, 1])
                
                angle = np.arccos(np.dot(lookat, last_cam_rotation))
                angle = np.degrees(angle)

                movement = np.sqrt(((C2W_T-last_cam_position)**2).sum())

                # print("angle:", angle, "movement:", movement)

                if angle>10 or movement>0.2:
                    last_cam_position = C2W_T
                    last_cam_rotation = lookat


                    new_camera = OnlineCam(
                        image_width,
                        image_height,
                        fov_y,
                        fov_x,
                        0.5,
                        10000.0,
                        cx, cy,
                        image,
                        R, T,
                        data_device="cpu",
                        id=len(scene.online_cameras)+1
                    )


                    

                    # scene.train_cameras[1.0].append(new_camera)
                    scene.online_cameras.append(new_camera)
                    camera_add_list.append(new_camera)
                    viewpoint_stack = viewpoint_stack + [new_camera for i in range(10)]
                    # viewpoint_indices = list(range(len(viewpoint_stack)))

            xyzs_list = np.concatenate(xyzs_list, axis=0).reshape(-1, 3)
            colors_list = np.concatenate(colors_list, axis=0).reshape(-1, 3)
            xyzs_list = torch.from_numpy(xyzs_list).to(device="cuda")
            colors_list = torch.from_numpy(colors_list).to(device="cuda")

            if gaussians._xyz.shape[0] > 0 and xyzs_list.shape[0] > 0:
                mask = add_points_with_voxel_limit(gaussians._xyz.data, xyzs_list, 0.1, 30)
                print(mask.sum(), xyzs_list.shape[0])
            else:
                mask = torch.ones(xyzs_list.shape[0], dtype=torch.bool, device="cuda")

            # num_samples = min(200000, xyzs_list.shape[0])
            # indices = torch.randperm(xyzs_list.shape[0], device="cuda")[:num_samples]
            # 使用相同索引同步采样
            # xyzs_list = xyzs_list[indices]
            # colors_list = colors_list[indices]
            # print(xyzs_list.shape, mask.sum())
            # xyzs_list = xyzs_list[mask]
            # colors_list = colors_list[mask]

            need_to_setup_training = False
            if gaussians._xyz.shape[0] == 0:
                need_to_setup_training = True
            if mask.sum() > 0:
                gaussians.add_pcd(xyzs_list, colors_list, 1.0)
                print("Added %d points" % xyzs_list.shape[0])
            torch.cuda.empty_cache()
            if need_to_setup_training:
                gaussians.training_setup(opt)

            
            for i, camera in enumerate(camera_add_list):
                gaussians.exposure_optimizer.add_param_group({"params": camera.exposure_param,"lr": 0.0001})
                gaussians.pose_optimizer.add_param_group({"params": [camera.cam_rot_delta, camera.cam_trans_delta], "lr": 0.00001})
            


        # 在没有收到任何训练图像的时候，死循环
        if scene.online_cameras == [] or gaussians._xyz.shape[0]==0:
            continue
        

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.online_cameras.copy()
        #     viewpoint_indices = list(range(len(viewpoint_stack)))
        # rand_idx = randint(0, len(viewpoint_indices) - 1)
        # viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx)
        if viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
        else:
            viewpoint_cam_index = randint(0, len(scene.online_cameras)-1)
            viewpoint_cam = scene.online_cameras[viewpoint_cam_index]

        viewpoint_cam.trained_iter+=1
        iteration+=1

        # # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # print(gaussians._xyz.isnan().any())
        # print(gaussians._features_dc.isnan().any())
        # print(gaussians._features_rest.isnan().any())
        # print(gaussians._opacity.isnan().any())
        # print(gaussians._scaling.isnan().any())
        # print(gaussians._rotation.isnan().any())
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = image*torch.exp(viewpoint_cam.exposure_param[:3].reshape(3, 1, 1)) + (viewpoint_cam.exposure_param[3:].reshape(3, 1, 1))
        

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # print(image.isnan().any(), viewpoint_cam.original_image.isnan().any())
        # window.view3.update_image((image.detach().cpu().numpy()*255).astype(np.uint8))
        # window.view4.update_image((gt_image.detach().cpu().numpy()*255).astype(np.uint8))
        import torchvision
        os.makedirs("debug", exist_ok=True)
        if iteration % 20 == 0:
            torchvision.utils.save_image(torch.concat([image, gt_image, torch.abs(image-gt_image)], dim=1), f"debug/debug_image_{iteration}.png")
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        opacity_reg = gaussians.get_opacity[render_pkg["visibility_filter"]].mean() * 0.000
        scaling_reg = gaussians.get_scaling.mean() * 0.01
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opacity_reg + scaling_reg

        # Depth regularization
        # Ll1depth_pure = 0.0
        # if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #     invDepth = render_pkg["depth"]
        #     mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #     depth_mask = viewpoint_cam.depth_mask.cuda()

        #     Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        #     Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        #     loss += Ll1depth
        #     Ll1depth = Ll1depth.item()
        # else:
        #     Ll1depth = 0

        loss.backward()

        

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Num": f"{gaussians._xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.pose_optimizer.step()
                gaussians.pose_optimizer.zero_grad(set_to_none = True)
                if viewpoint_cam.trained_iter > 3:
                    update_pose(viewpoint_cam)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
