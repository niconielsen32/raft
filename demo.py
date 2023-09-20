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

import numpy as np
import pyzed.sl as sl
import cv2



DEVICE = 'cuda'


def RAFT_stereo_live(args):
    
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    
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
    
    
    # Calibration
    fx, fy, cx1, cy = 3997.684, 3997.684, 1176.728, 1011.728
    cx2 = 1307.839
    baseline=193.001 # in millimeters

    

    with torch.no_grad():
        while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Grab an image, a RuntimeParameters object must be given to grab()
            
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left = image_left.get_data()[:,:,:3]
            right = image_right.get_data()[:,:,:3]
            
            #print(left.shape, right.shape)
            
            left_downscaled = cv2.resize(left, (640, 480))
            right_downscaled = cv2.resize(right, (640, 480))
        
            left_downscaled = torch.from_numpy(left_downscaled).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
            right_downscaled = torch.from_numpy(right_downscaled).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
            

            padder = InputPadder(left_downscaled.shape, divis_by=32)
            left_downscaled, right_downscaled = padder.pad(left_downscaled, right_downscaled)

            _, flow_up = model(left_downscaled, right_downscaled, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            disparity = flow_up.cpu().numpy().squeeze()
            
            disparity = -disparity
            
            print(disparity.shape)
            
            # Disparity to point cloud
            # inverse-project
            depth = (fx * baseline) / (disparity + (cx2 - cx1))
            H, W = depth.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

            mask = np.ones((H, W), dtype=bool)

            # Remove flying points
            #mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
            #mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

            points = points_grid.transpose(1,2,0)[mask]
            colors = left_downscaled[mask].astype(np.float64) / 255
            
            
            max_value = np.max(disparity)
            min_value = np.min(disparity)
            print(f"Max: {max_value}, Min: {min_value}")
            
            NUM_POINTS_TO_DRAW = 100000

            subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=True)
            points_subset = points[subset]
            colors_subset = colors[subset]
            
            x, y, z = points.T
            
            
            if args.save_numpy:
                np.save(f"points.npy", disparity)
                
                
            disparity = cv2.resize(disparity.astype(np.int8), (1920,1080), interpolation=cv2.INTER_NEAREST)
            
            #left = cv2.resize(disparity.astype(np.int8), (1920,1080), interpolation=cv2.INTER_NEAREST)
            #right = cv2.resize(disparity.astype(np.int8), (1920,1080), interpolation=cv2.INTER_NEAREST)
                
            cv2.imshow("Mask", mask.astype(np.int8))
            cv2.imshow("Disparity", cv2.resize(disparity.astype(np.int8), (4416, 1242), interpolation=cv2.INTER_NEAREST))
            
            
            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        # Close the camera
        zed.close()
        cv2.destroyAllWindows()

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
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
