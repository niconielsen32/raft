import numpy as np
import torch
import cv2
import pyzed.sl as sl


from RAFT_DEPTH_STEREO import load_model, generate_depth

model = load_model()

depth = generate_depth(model, left, right)