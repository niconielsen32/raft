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



def load_model(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model



def generate_depth(model, left, right):
    
    h, w, c = left.shape
    
    print(left.shape)
    
    left = cv2.resize(left, (640, 480))
    right = cv2.resize(right, (640, 480))
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    image1 = torch.from_numpy(left).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    image2 = torch.from_numpy(right).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
    flow_up = padder.unpad(flow_up).squeeze()

    disparity = flow_up.cpu().numpy().squeeze()
    
    disparity = -disparity
    
    disparity = cv2.resize(disparity, (w, h))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    left = cv2.resize(left, (w, h))
    
    
    
    # Disparity to point cloud
    # inverse-project

    # Calibration
    fx, fy, cx1, cy = 3997.684, 3997.684, 1176.728, 1011.728
    cx2 = 1307.839
    baseline=193.001 # in millimeters
    
    depth = (fx * baseline) / (disparity + (cx2 - cx1))

    H, W = depth.shape
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
    
    return left, depth, disparity, projected_points, colors, x, y, z



def RAFT_stereo_live(args):
    
    model = load_model(args)
    
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)


    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    first = True

    with torch.no_grad():
        while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Grab an image, a RuntimeParameters object must be given to grab()
            
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left = image_left.get_data()[:,:,:3]
            right = image_right.get_data()[:,:,:3]
            
            #print(left.shape, right.shape)
            
            left, depth, disparity, projected_points, colors, x, y, z = generate_depth(model, left, right)
            
            print(x.shape, y.shape, z.shape)
            print(projected_points.shape)
            
            #print(x.shape, y.shape, z.shape)
            
            #print(np.allclose(points[:,:3], projected_points))
            
            """if first:
                
                f = go.Figure(data=[
                            go.Scatter3d(
                            x=-x, y=-z, z=-y, # flipped to make visualization nicer
                            mode='markers',
                            marker=dict(size=1, color=colors))],
                            layout=dict(scene=dict(
                            xaxis=dict(visible=True),
                            yaxis=dict(visible=True),
                            zaxis=dict(visible=True),)))
                first = False
                f.show()
               
            else:
                f.data[0].x = x
                f.data[0].y = -z
                f.data[0].z = -y
                f.data[0].marker.color = colors
                #f.write_html("temp-plot.html", include_plotlyjs="cdn", auto_open=False)"""
        
            if args.save_numpy:
                np.save(f"points.npy", projected_points)
                
                
            cv2.imshow("Disparity", cv2.resize(disparity.astype(np.int8), (1920,1080), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("Left", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imshow("Right", right)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            

        # Close the camera
        zed.close()
        cv2.destroyAllWindows()

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="models/raftstereo-eth3d.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    RAFT_stereo_live(args)

