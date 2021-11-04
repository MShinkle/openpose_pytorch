conda create -n openpose_pytorch -y python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openpose_pytorch
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y ipykernel
conda install -y matplotlib
pip install wget
