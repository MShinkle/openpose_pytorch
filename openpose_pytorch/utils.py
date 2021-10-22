import torch
import cv2
from torchvision import transforms
from torchvision.transforms import functional
import numpy as np

def load_and_transform(files, resolution=(500,500)):
    files_array = np.array([cv2.imread(file) for file in files]).transpose(0,3,1,2)
    vertical, horizontal = files_array.shape[2:]
    vpad, hpad = (max((horizontal-vertical)//2,0), max((vertical-horizontal)//2,0))

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Pad((hpad, vpad)),
        transforms.Resize(resolution),
    ])

    files_array = np.array([cv2.imread(file) for file in files]).transpose(0,3,1,2)
    return transform(torch.tensor(files_array)).to('cuda')

def upscale_unpad(padded_tensor, original_dims):
    padded_size = padded_tensor.shape[2:]
    max_dim = max(original_dims)
    vertical_padding = int(max(original_dims[0] - original_dims[1], 0)/2)
    horizontal_padding = int(max(original_dims[1] - original_dims[0], 0)/2)
    upscaled = functional.resize(padded_tensor, (max_dim, max_dim))
    unpadded = upscaled[vertical_padding:vertical_padding+original_dims[0], horizontal_padding:horizontal_padding+original_dims[1]]
    return unpadded