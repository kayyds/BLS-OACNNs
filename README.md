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
conda create -n bls_env python=3.8 -y
conda activate bls_env
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install spconv-cu118
pip install -r requirements.txt

## 📊 Data Preparation
We evaluate our model on the **ScanNet v2** dataset. 
Please follow the standard Pointcept data preparation pipeline to preprocess the point cloud data and place the processed dataset in the `./data/scannet` folder.
