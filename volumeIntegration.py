
import torch
import cv2

import argparse
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import pyzed.sl as sl


import open3d as o3d

import time



    
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


calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

focal_left_x = calibration_params.left_cam.fx
focal_left_y = calibration_params.left_cam.fy
cx = calibration_params.left_cam.cx
cy = calibration_params.left_cam.cy

intrinsics = np.array([[focal_left_x, 0, cx], [0, focal_left_y, cy], [0, 0, 1]]).astype(np.float32)
intrinsics = np.ascontiguousarray(intrinsics).astype(np.float32)

print(intrinsics)




def generate_point_cloud(image, depth):
     # Calibration
    #fx, fy, cx1, cy = 3000, 3000, 512, 512
    
    #fx, fy, cx1, cy = 1076.4019775390625, 1076.4019775390625, 980.2750244140625, 548.6805419921875
    #cx2 = 1307.839
    #baseline=193.001 # in millimeters
    fx = 1076.4019775390625
    fy = 1076.4019775390625
    cx1 = 980.2750244140625
    cx2 = 0
    baseline=120.001 # in millimeters
    
    depth = (fx * baseline) / (-depth + (cx2 - cx1))
    
    
    #inverse depth
    #depth = 1 - depth
    
    #depth = cv2.resize(disparity, (w, h))
    
    max_depth = np.max(depth)
    min_depth = np.min(depth)
    
    print("max_depth: ", max_depth)
    print("min_depth: ", min_depth)
    
    #depth = cv2.resize(depth, (w, h))

    H, W = depth.shape[:2]
    
    print(depth.shape)

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    print(np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0).shape)
    points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth.transpose(2, 0, 1)

    mask = np.ones((H, W), dtype=bool)

    # Remove flying points
    #mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    #mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

    points = points_grid.transpose(1,2,0)[mask]
    colors = image[mask].astype(np.float64) / 255
    
    
    return points, colors





img0 = cv2.imread('img0.png')
depth0 = cv2.imread('depth_norm0.png', cv2.IMREAD_UNCHANGED)

print(depth0.dtype)
#depth0 = cv2.resize(depth0, (1920, 1080))

img1 = cv2.imread('img1.png')
depth1 = cv2.imread('depth_norm1.png', cv2.IMREAD_UNCHANGED)
#depth1 = cv2.resize(depth1, (1920, 1080))

print(img0.shape)
print(depth0.shape)
cv2.imshow('img0', img0)
cv2.imshow('depth_norm0', depth0)
cv2.imshow('img1', img1)
cv2.imshow('depth_norm1', depth1)
cv2.waitKey(0)


points, colors = generate_point_cloud(img0, depth0)
points = points.astype(np.float64)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])

points, colors = generate_point_cloud(img1, depth1)
points = points.astype(np.float64)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points)
pcd2.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd2])



img00 = o3d.geometry.Image(img0)
img11 = o3d.geometry.Image(img1)
depth00 = o3d.geometry.Image(depth0)
depth11 = o3d.geometry.Image(depth1)

source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img00, depth00, depth_trunc=np.inf, convert_rgb_to_intensity=False)
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics)))

print(source_pcd)
o3d.visualization.draw_geometries([source_pcd])


target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img11, depth11, depth_trunc=np.inf, convert_rgb_to_intensity=False)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics)))

print(target_pcd)
o3d.visualization.draw_geometries([target_pcd])



option = o3d.pipelines.odometry.OdometryOption()
odo_init = np.identity(4).astype(np.float64)
print(option)

print("Apply point-to-plane ICP")
[success_color_term, trans_color_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics)),
     jacobian=o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm())

print("Apply hybrid RGB-D ICP")

intrinsic_matrix = o3d.core.Tensor( 
     intrinsics, 
     dtype=o3d.core.Dtype.Float32) 

[success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics)),
     jacobian=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())

print("Color term:")



print(success_color_term)
print(success_hybrid_term)

if success_color_term:
    print("Using RGB-D Odometry")
    print(trans_color_term)
    #source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsics)
    #points, colors = generate_point_cloud(img0, depth0)
   #points = points.astype(np.float64)
    source_pcd.transform(trans_color_term)
    
    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    
if success_hybrid_term:
    print("Using Hybrid RGB-D Odometry")
    print(trans_hybrid_term)
    #source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsics)

    #pcd.transform(trans_hybrid_term)
    #o3d.visualization.draw_geometries([pcd2, pcd])
    
    
    
print("Integrate the source RGB-D image into the volume.")
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.001,
    sdf_trunc=0.002,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


print("first integration")
volume.integrate(
     source_rgbd_image,
     o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics),
     np.linalg.inv(np.identity(4)))


print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
pcd = volume.extract_point_cloud()
mesh.compute_vertex_normals()
print(pcd)
o3d.visualization.draw_geometries([pcd])


print("second integration")
volume.integrate(
     target_rgbd_image,
     o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, intrinsic_matrix=intrinsics),
     np.linalg.inv(trans_hybrid_term))



print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
pcd = volume.extract_point_cloud()
mesh.compute_vertex_normals()
print(pcd)
o3d.visualization.draw_geometries([pcd])
