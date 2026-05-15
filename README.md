# FD-UNet: A Frequency Decoupled U-Net for Building Segmentation in High-Resolution Remote Sensing Imagery

This is an official PyTorch implementation of "[**FD-UNet: A Frequency Decoupled U-Net for Building Segmentation in High-Resolution Remote Sensing Imagery**]".

# Introduction
Accurate building segmentation in high-resolution (HR) remote sensing imagery is fundamentally challenged by the complex interweaving of high-frequency structural boundaries and low-frequency semantic regions. Traditional spatial-domain convolution filters often fail to adaptively decouple these distinct features, leading to common failure modes such as boundary blurring and segmentation fragmentation. To address this frequency-entanglement dilemma, we propose FD-UNet, a novel Frequency Decoupled U-Net architecture for high-fidelity building extraction. Specifically, we propose the Frequency Dynamic ConvNeXt (FDCNX) encoder, which incorporates Frequency Dynamic Convolution (FDC) to explicitly decouple and adaptively extract distinct frequency bands. This mechanism significantly enhances the precision of boundary delineation and roof texture representation without increasing parameter redundancy. To ensure effective feature reconstruction, a Multi-dimensional Hybrid Attention with Residual (MHAR) decoder is developed. By synergistically leveraging a Dual-branch Linear Attention Block (DLAB) for semantic bridging and a Multi-scale Channel Spatial Attention Block (MCSAB) for noise suppression, the decoder progressively purifies cross-stage features during the upsampling process.
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

