
import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import pyzed.sl as sl
import cv2

import plotly.graph_objects as go
import plotly.express as px

import open3d as o3d

import time



def load_model(restore_ckpt="models/raftstereo-realtime.pth", save_numpy=False, mixed_precision=True, valid_iters=12, hidden_dims=[128]*3, corr_implementation="reg_cuda", shared_backbone=True, corr_levels=4, corr_radius=4, n_downsample=3, context_norm="batch", slow_fast_gru=True, n_gru_layers=2):
    
    model = torch.nn.DataParallel(RAFTStereo(restore_ckpt, save_numpy, mixed_precision, valid_iters, hidden_dims, corr_implementation, shared_backbone, corr_levels, corr_radius, n_downsample, context_norm, slow_fast_gru, n_gru_layers), device_ids=[0])
    model.load_state_dict(torch.load(restore_ckpt))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    return model



def generate_depth(model, left, right):
    
    h, w, c = left.shape
    
    downscale_factor = 0.7
    
    down_width = int(640*downscale_factor)
    down_height = int(480*downscale_factor)
    
    width_ratio = w / down_width
    height_ratio = h / down_height
    
    left = cv2.resize(left, (down_width, down_height))
    right = cv2.resize(right, (down_width, down_height))
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    image1 = torch.from_numpy(left).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    image2 = torch.from_numpy(right).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    _, flow_up = model(image1, image2, iters=12, test_mode=True)
    flow_up = padder.unpad(flow_up).squeeze()

    disparity = flow_up.cpu().numpy().squeeze()
    
    disparity = -disparity
    
    #disparity = cv2.resize(disparity, (w, h))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    #left = cv2.resize(left, (w, h))
    
    
    # Disparity to point cloud
    # inverse-project

    # Calibration
    fx, fy, cx1, cy = 3997.684, 3997.684, 1176.728, 1011.728
    cx2 = 1307.839
    baseline=193.001 # in millimeters
    baseline=120.001 # in millimeters
    
    depth = (fx * baseline) / (-disparity + (cx2 - cx1))
    
    #inverse depth
    #depth = 1 - depth
    
    #depth = cv2.resize(disparity, (w, h))
    
    depth = cv2.resize(depth, (w, h))

    """H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

    mask = np.ones((H, W), dtype=bool)

    # Remove flying points
    #mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    #mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

    points = points_grid.transpose(1,2,0)[mask]
    colors = left[mask].astype(np.float64) / 255
    
    
    # Calculate the half angle of the field of view
    fov = 110 # in degrees
    half_angle = fov / 2.0 * np.pi / 180.0

    # Calculate the aspect ratio of the image
    aspect_ratio = W / H

    # Calculate the distance from the camera to the scene
    # You can use any point in the scene, but for simplicity, we use the center point
    center_depth = depth[H//2, W//2]

    # Calculate the top and right edges of the orthogonal projection
    top = center_depth * np.tan(half_angle)
    right = top * aspect_ratio
    
    far = 1
    near = 0
    
    
    def ortho(left, right, bottom, top, near, far):
        # Create an orthographic projection matrix using numpy
        # Args:
        # left: float, the left clipping plane
        # right: float, the right clipping plane
        # bottom: float, the bottom clipping plane
        # top: float, the top clipping plane
        # near: float, the near clipping plane
        # far: float, the far clipping plane
        # Returns:
        # ortho_matrix: ndarray of shape (4, 4), the orthographic projection matrix
        ortho_matrix = np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                                [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                                [0, 0, -2/(far-near), -(far+near)/(far-near)],
                                [0, 0, 0, 1]])
        return ortho_matrix
    


    # Create an orthogonal projection matrix using numpy
    ortho_matrix = np.array([[1/right, 0, 0, 0],
                            [0, 1/top, 0, 0],
                            [0, 0, -2/(far-near), -(far+near)/(far-near)],
                            [0, 0, 0, 1]])
    
    ortho_matrix = ortho(-right, right, -top, top, near, far)
    
    # Add a fourth component of 1 to the points
    points = np.append(points, np.ones((points.shape[0], 1)), axis=1)

    # Multiply the orthogonal projection matrix by the points
    projected_points = np.dot(ortho_matrix, points.T).T

    # Divide by the fourth component to get the normalized coordinates
    projected_points = projected_points / points[:, 3].reshape(-1, 1)

    
    max_value = np.max(disparity)
    min_value = np.min(disparity)
    #print(f"Max: {max_value}, Min: {min_value}")
    
    NUM_POINTS_TO_DRAW = 50000

    subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=True)
    points_subset = points[subset]
    colors_subset = colors[subset]
    
    #print(projected_points.shape)
    projected_points = projected_points[:,:3]
    x, y, z = projected_points.T
    
    # return everything
    #return left, depth, disparity, projected_points, colors, x, y, z
    # Only return depth"""
    return depth



def RAFT_stereo_live():
    
    model = load_model()
    
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)


    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    depth_for_display = sl.Mat()
    
    #calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    #focal_left_x = calibration_params.left_cam.fx
    #focal_left_y = calibration_params.left_cam.fy
    #cx = calibration_params.left_cam.cx
    #cy = calibration_params.left_cam.cy
    
    #print(focal_left_x, focal_left_y, cx, cy)

    with torch.no_grad():
        while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Grab an image, a RuntimeParameters object must be given to grab()
            
            start = time.perf_counter()
            
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)

            left = image_left.get_data()[:,:,:3]
            right = image_right.get_data()[:,:,:3]
            depth_map = depth_for_display.get_data()
            
            #left, depth, disparity, projected_points, colors, x, y, z = generate_depth(model, left, right)
            depth = generate_depth(model, left, right)
            depth_normalized = cv2.normalize(depth, None, 0, 255.0, cv2.NORM_MINMAX)
            
            end = time.perf_counter()
            total = end - start
            fps = 1 / total
            
            
            left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            cv2.putText(left, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            cv2.imshow("Depth", depth_map)
            #print(depth.dtype)
            cv2.imshow("Geneated Depth", depth_normalized.astype(np.uint8))
            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            

        # Close the camera
        zed.close()
        cv2.destroyAllWindows()

            


if __name__ == '__main__':

    RAFT_stereo_live()
