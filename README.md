# DMAConv
DMAConv: Dual Mask-Adaptive Convolution for Remote Sensing Pansharpening

## Clone this repository
```bash
git clone https://github.com/Xiao-Xianghong/DMAConv.git
cd DMAConv
```

## Environment and dependence
You're supposed to use a conda virtual environment. 

- Python 3.10
- PyTorch 2.5.1
- CUDA 13.1

Install PyTorch according to your CUDA version:
https://pytorch.org/get-started/locally/

run this code in bash to install dependences (include torch==2.5.1)
```bash
pip install -r requirements.txt
```

## Dataset
We use the Tiny Shakespeare dataset, a widely used and well-established dataset for language modeling tasks.

## Training Details
The batch size is set to 4 due to GPU memory constraints (NVIDIA GeForce RTX 4060 Laptop GPU with 8GB VRAM).

If you have access to a GPU with larger memory, increasing the batch size may help accelerate training.

The model is trained for 500 epochs, and checkpoints are saved every 250 epochs.

## Start training！
```bash
python train.py
```
