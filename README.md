# BLS-OACNNs

[![DOI](https://zenodo.org/badge/xxxxxx.svg)](https://zenodo.org/badge/latestdoi/xxxxxx)

This repository contains the official PyTorch implementation and core experimental data for the paper:

**"Hybrid Broad Learning for Efficient 3D Semantic Segmentation: A Lightweight Approach"**

> **News:** This manuscript is currently under review for publication in ***The Visual Computer*** (Springer Nature). 

## 📌 Citation
If you find our work, algorithm, or data helpful in your research, please consider citing our manuscript:
```bibtex
@article{BLS_OACNNs_2026,
  title={Hybrid Broad Learning for Efficient 3D Semantic Segmentation: A Lightweight Approach},
  author={Zhang, Guoyou and Kang, Ang and Pan, Lihu and Li, Yunhao and Zhang, Chen and Guo, Ziyang and Wang, Jian},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}
```

## 📖 Methodology / Key Algorithms
To address the quadratic computational bottleneck of attention mechanisms in 3D perception, we propose **BLS-OACNNs**. The core innovation lies in the **BLS-Hybrid Stage strategy**, which systematically substitutes compute-intensive attention modules with our efficient pseudo-inverse-based **BLSBlocks**. This establishes a Native Efficient Design paradigm that scales linearly with scene size while maintaining high segmentation accuracy.

## 🛠️ Environment Setup & Dependencies
We recommend using Conda for environment setup. Core dependencies include:
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8
- `spconv` (Sparse Convolution)

**Example Conda Setup:**
```bash
conda create -n bls_env python=3.8 -y
conda activate bls_env
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install spconv-cu118
pip install -r requirements.txt
```

## 📊 Data Preparation
We evaluate our model on the **ScanNet v2** dataset. 
Please follow the standard Pointcept data preparation pipeline to preprocess the point cloud data and place the processed dataset in the `./data/scannet` folder.

## 🚀 Quick Start (Training & Evaluation)

### Training
To train the BLS-OACNNs model on ScanNet v2 from scratch, run the following command:
```bash
sh scripts/train.sh -p python -d scannet -c semseg-bls-oacnns-v1m1-0-base -n bls_oacnns_run
```

### Evaluation / Testing
To evaluate the trained model and reproduce our results, use the testing script:
```bash
sh scripts/test.sh -p python -d scannet -c semseg-bls-oacnns-v1m1-0-base -n bls_oacnns_run -w model_best
```

## 📄 License
This project is released under the MIT License.
```
