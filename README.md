# FDCNX-UNet Frequency Dynamic ConvNeXt UNet with Linear Attention for Building Segmentation

This is an official PyTorch implementation of "[**FDCNX-UNet Frequency Dynamic ConvNeXt UNet with Linear Attention for Building Segmentation**]".

# Introduction
with the rapid development of remote sensing technology and deep learning, significant progress has been made in building segmentation from remote sensing images using semantic segmentation models. Currently, the encoder-decoder architecture is the most widely adopted framework in building segmentation research, though challenges such as blurred boundaries and poor segmentation performance for small targets persist. This is primarily due to network limitations in encoder feature extraction capability and decoder processing of transmission features. To address these challenges, we propose an encoder-decoder network named FDCNX-UNet capable of capturing global context.
<center> 
<img src="DRAU-Net.png" width="auto" height="auto">
</center>

# Image segmentation

## 1. Requirements
```
# Environments:
cuda==11.8
python==3.9
# Dependencies:
pip install torch==2.0.0 torchvision==0.22.1
pip install einops==0.6.1 imageio==2.28.1   albumentations   Torchmetrics==0.11.4
```

## 2. Data Preparation

```
│inria/
├──austin1/
│  ├── images
│  │   ├── austin1.jpg
│  │   ├── ......
│  ├── binary_masks
├──austin2/
│  ├── images
│  │   ├── austin2.jpg
│  │   ├── ......
│  ├── ......
```

## 3. Train

python train.py --dataset inria 

## 4. Validation

python test.py --load_checkpoint output/checkpoints/20250401-0413_unet/20250401-0413_unet_e1.pt

