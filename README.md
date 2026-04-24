# SSDN: Spectral-Spatial Dual-Domain Synergistic Network

> **Dual-Domain Super-Resolution with Physical Degradation Modeling for Enhancing Micro-Defect Detectability in VCSELs**

This repository contains the official PyTorch implementation of the **SSDN** framework for semiconductor electroluminescence (EL) image super-resolution. 

Our method decouples structural restoration from noise suppression by synergistically combining a Coordinate-Attention Spatial Feature Enhancement Module (**CASFEM**) and a Dynamic Frequency-domain Interaction Block (**DFIB**), achieving state-of-the-art performance in recovering micron-scale defects under non-ideal industrial imaging conditions.

---

## 🚨 Disclaimer (Data & Weights)
Due to the **Non-Disclosure Agreement (NDA)** with our industrial semiconductor manufacturing partners, the proprietary VCSEL electroluminescence dataset and the pre-trained model weights cannot be made publicly available. 

This repository provides the complete **Architecture Implementation**, **Training Pipeline**, and **Evaluation Scripts** to demonstrate the exact methodology and parameter calculations presented in the paper. The scripts can be adapted to your own custom datasets.

---

## 📁 Repository Structure

```text
SSDN-Super-Resolution/
├── datasets/               # Placeholder for custom datasets (Train/Test)
│   └── README.md           # Instructions for data organization
├── experiments/
│   └── ssdn_weights/       # Placeholder for model weights & NDA disclaimer
│       └── README.md
├── results/                # Placeholder for super-resolved output images
│   └── README.md
├── data/
│   ├── __init__.py
│   └── dataset.py          # Custom dataset loader with geometric augmentations
├── models/
│   ├── __init__.py
│   └── ssdn.py             # Core architecture: SSDN, CASFEM, and DFIB
├── scripts/
│   ├── train.py            # Distributed Data Parallel (DDP) training script
│   └── test.py             # Evaluation script with rigorous boundary cropping & Y-channel PSNR/SSIM
├── utils/
│   ├── __init__.py
│   └── metrics.py          # Standardized metric calculations
├── requirements.txt        # Environment dependencies
└── README.md               # Project overview and usage guide
```
## ⚙️ Installation

The code has been tested on **Ubuntu 20.04** with **Python 3.9** and dual **NVIDIA GeForce RTX 5090** GPUs.
1. Clone the repository
2. Install the required dependencies: pip install -r requirements.txt

## 🚀 Usage

### 1. Distributed Training (DDP)
The training pipeline is optimized for a multi-GPU environment using PyTorch Distributed Data Parallel (DDP). To launch training on a dual-GPU setup, run:

```bash
torchrun --nproc_per_node=2 scripts/train.py \
    --lr_train_path /path/to/your/train_LR \
    --hr_train_path /path/to/your/train_HR \
    --batch_size  \
    --scale  \
    --epochs  \
    --checkpoint_dir experiments/ssdn_weights
```
### 2. Evaluation & Metric Calculation
To evaluate the model performance, run the following script. It performs inference and calculates PSNR/SSIM on the Y-channel with boundary cropping (shaving) matching the scale factor:
```bash
python scripts/test.py \
    --model_path experiments/ssdn_weights/ssdn_epoch_500.pth \
    --lr_dir /path/to/your/test_LR \
    --hr_dir /path/to/your/test_HR \
    --scale  \
    --in_chans  \
    --save_dir results/
```
