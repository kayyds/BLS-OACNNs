# BLS-OACNNs
Official PyTorch implementation of "BLS-OACNNs: A Lightweight Hybrid-Module Network
# BLS-OACNNs: A Lightweight Hybrid-Module Network for Efficient 3D Semantic Segmentation

This repository contains the official PyTorch implementation of the paper **"BLS-OACNNs: A Lightweight Hybrid-Module Network for Efficient 3D Semantic Segmentation"** (Under Review).

Our code is built upon the [Pointcept](https://github.com/Pointcept/Pointcept) framework.

## 1. Environment Setup
Please refer to `scripts/build_image.sh` for Docker environment setup, or install the required dependencies manually (PyTorch, spconv, etc.) as standard Pointcept requirements.

## 2. Data Preparation
We evaluate our model on the ScanNet v2 dataset. Please follow the standard Pointcept data preparation pipeline to preprocess the dataset and place it in the `data/scannet` folder.

## 3. Training
To train the BLS-OACNNs model on ScanNet v2 from scratch, run the following command:
```bash
sh scripts/train.sh -p python -d scannet -c semseg-bls-oacnns-v1m1-0-base -n bls_oacnns_run
