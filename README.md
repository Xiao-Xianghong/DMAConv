# DMAConv
DMAConv: Dual Mask-Adaptive Convolution for Remote Sensing Pansharpening

## Clone this repository
```bash
git clone https://github.com/Xiao-Xianghong/DMAConv.git
cd DMAConv
```

## Environment and dependence
You're supposed to use a conda virtual environment. 

- Python 3.10 with dev
- PyTorch 2.5.1 + cu118
- CUDA 13.1

Install PyTorch according to your CUDA version:
https://pytorch.org/get-started/locally/

run this code in bash to install dependences (include torch==2.5.1)
```bash
pip install -r requirements.txt
```

## Install PWAC
Pixel-Wise Adaptive Convolution is needed in this project

Before installing, please make sure that CMake (version 3.26 or later) and Ninja are installed on your system and properly configured in your environment variables.

'''bash
conda activate yourenv
git clone https://github.com/src-d/kmcuda
cd src
cmake -DCMAKE_BUILD_TYPE=Release . && make
cd ..
bash ./build.sh
'''

## Dataset
The training and test datasets used in this model are from the public dataset ''PanCollection'', which can be downloaded from the following URL: https://liangjiandeng.github.io/PanCollection.html

## Start training！
```bash
python train.py
```
