import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import sys
import numpy as np
import pyzed.sl as sl
import cv2
from shutil import which



model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
model.load_state_dict(torch.load(args.restore_ckpt))

DEVICE = 'cuda'
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




while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # Grab an image, a RuntimeParameters object must be given to grab()
    
    zed.retrieve_image(image_left, sl.VIEW.LEFT)
    zed.retrieve_image(image_right, sl.VIEW.RIGHT)
    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured

    image_ocv_left = image_left.get_data()
    image_ocv_right = image_right.get_data()
    #cv2.imwrite("left_zed.png", image_ocv_left)
    #cv2.imwrite("right_zed.png", image_ocv_right)
    
    image_ocv_left = cv2.resize(image_ocv_left, (640,480))
    image_ocv_right = cv2.resize(image_ocv_right, (640,480))
    
    image_ocv_left_gray = cv2.cvtColor(image_ocv_left, cv2.COLOR_BGR2GRAY)
    image_ocv_right_gray = cv2.cvtColor(image_ocv_right, cv2.COLOR_BGR2GRAY)
    
    
    cv2.imshow("left", image_ocv_left)
    cv2.imshow("right", image_ocv_right)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Close the camera
zed.close()
cv2.destroyAllWindows()

