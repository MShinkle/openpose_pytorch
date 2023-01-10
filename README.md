# openpose_pytorch

Batch processing of images for pose estimation via OpenPose (currently only body keypoints).  Implemented in PyTorch.

Based heavily on https://github.com/Hzzone/pytorch-openpose, but with simplification, support for batch processing and conda-based setup.

# Setup

To create a compatible anaconda environment, clone this repo and run `bash setup_openpose_env.sh` in the directory.

Alternatively, its contents can be run line-wise at the command line:

```
conda create -n openpose_pytorch -y python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openpose_pytorch
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y ipykernel
conda install -y matplotlib
pip install wget
```

Activate the environment if your not already in it (`conda activate openpose_pytorch`) and run the setup script (`python3 setup.py install`).

Note: has not been tested in Windows or macOS.

# Basic usage

```
import numpy as np
import torch
from openpose_pytorch.models import body_pose
from openpose_pytorch.utils import load_and_transform, upscale_unpad
from openpose_pytorch.keypoints import get_candidates_subsets, get_keypoints
import glob
from matplotlib import pyplot as plt

model = body_pose(pretrained=True)
images_list = sorted(glob.glob('/path/to/images/directory/*'))
images = load_and_transform(images_list, resolution=(250,250))
if torch.cuda.is_available():
    model = model.cuda()
    images = images.cuda()

pafs, heatmaps = model(images)
orig_size = plt.imread(images_list[0]).shape[:2]
pafs = upscale_unpad(pafs, orig_size)
heatmaps = upscale_unpad(heatmaps, orig_size)
candidates, subsets = get_candidates_subsets(heatmaps, pafs)
keypoints = get_keypoints(candidates, subsets)
```
This will return a list of n_people x 18 x 3 keypoints tensors for each input images.

# TODO
The majority of computational time is in the get_candidates_subsets step.  Refactoring could substantially improve runtime.

face and hand keypoints

higher-level wrapper

detection of isolated parts (e.g. hands + arm)

segmentation approximation
