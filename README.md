
<div align="center">
<h1>Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views</h1>

[**Jiang Wu**]() · [**Rui Li**](https://ruili3.github.io/) · [**Yu Zhu**]() · [**Rong Guo**]() · [**Jinqiu Sun**]() · [**Yanning Zhang**]()

Northwestern Polytechnical University

**CVPR 2025**

<a href='https://arxiv.org/abs/2504.20378'><img src='https://img.shields.io/badge/arXiv-Sparse2dgs-red' alt='arXiv'></a>

---

**⭐ This is an improved fork with enhanced features and optimizations for practical deployment. See [Improvements](#-improvements-over-original) below.**

**Original Repository:** [Wuuu3511/Sparse2DGS](https://github.com/Wuuu3511/Sparse2DGS)

</div>

![](assets/t.jpg)
## ⭐ Improvements Over Original

This fork includes significant enhancements for practical deployment and research reproduction:

### 🔧 Performance & Resource Optimization
- **Memory Optimization (12GB VRAM / 16GB RAM compatible)**
  - Automatic image downsampling to 720p for memory-constrained environments
  - Intelligent point cloud subsampling (max ~3M points) to prevent OOM
  - Delayed GPU loading - images stay on CPU until needed
  - FP16 storage for intermediate features (50% RAM reduction)

- **Training Speed Optimization**
  - Disabled expensive cross-view reprojection loss (10x speedup in large scenes)
  - Local neighborhood MVS (4 nearest views instead of 100+ images)
  - Dynamic resolution scaling for MVS phase (max 960px)
  - Explicit VRAM cleanup after each frame

### 🌐 Enhanced Compatibility
- **Camera Model Support**
  - OPENCV, SIMPLE_RADIAL, RADIAL models (original only supported PINHOLE)
  - Direct COLMAP compatibility - no need for camera model conversion
  - Automatic DTU format camera file generation with depth_min/max

- **Platform & Environment**
  - WSL2 support with automatic CUDA path injection
  - Updated COLMAP parameters (--FeatureExtraction.use_gpu)
  - GLOMAP (Global Mapper) integration for robust reconstruction
  - CUDA 12.x compatibility

### 🐛 Bug Fixes
- Fixed dimension mismatch in CLMVSNet (32-multiple alignment)
- Fixed feature map spatial alignment issues
- Fixed IndexError in point cloud selection
- Fixed NameError in loss calculation
- Fixed device mismatch (CPU/GPU mixed operations)

### 🎬 New Features
- **Video-to-Sparse2DGS Pipeline** (`scene2sparse2dgs/`)
  - Complete pipeline from mobile video to 3D reconstruction
  - Smart blurry image filtering (Laplacian variance)
  - Uniform sampling (max 300 images)
  - Integrated COLMAP + GLOMAP workflow
  - Automated error handling and recovery

- **Enhanced Visualization**
  - Feature map visualization utilities
  - PCA visualization with fixed dimension issues

See commit history for detailed changes: [fed2d61](https://github.com/tianxingleo/Sparse2DGS/commit/fed2d61)

---

## ☁ Introduction
This paper proposes Sparse2DGS, a Gaussian Splatting method tailored for surface reconstruction from sparse input views. Traditional methods relying on dense views and SfM points struggle under sparse-view conditions. While learning-based MVS methods can produce dense 3D points, directly combining them with Gaussian Splatting results in suboptimal performance due to the ill-posed nature of the geometry. Sparse2DGS addresses this by introducing geometry-prioritized enhancement schemes to enable robust geometric learning even in challenging sparse settings.
![](assets/arc.jpg)


## ⛏ Installation

### Basic Installation
```bash
conda create -n sparse2dgs python=3.8 # use python 3.8
conda activate sparse2dgs
pip install -r requirements.txt
```

We perform feature splatting by simply adjusting the number of rendering channels in the original "[diff_surfel_rasterization](https://github.com/hbb1/diff-surfel-rasterization/tree/e0ed0207b3e0669960cfad70852200a4a5847f61)".
```bash
python -m pip install setuptools==69.5.1
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

### Optional: For Video-to-Sparse2DGS Pipeline
```bash
# Install COLMAP (CUDA-enabled recommended)
# Ubuntu/WSL2:
sudo apt-get install colmap

# Or build from source for latest features:
# https://github.com/colmap/colmap/blob/main/src/colmap/CMakeLists.txt

# Install FFmpeg
sudo apt-get install ffmpeg
```

### WSL2 Users

If you're using WSL2, the pipeline will automatically inject CUDA paths. However, ensure:
- Windows CUDA drivers are installed
- COLMAP with CUDA support is available in WSL2

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM
- RAM: 16GB

**Recommended:**
- GPU: 12GB+ VRAM (RTX 3060 or better)
- RAM: 32GB
- For large scenes: 16GB+ VRAM


## 🎬 Video to Sparse2DGS Pipeline

**NEW:** Complete pipeline from mobile video to 3D reconstruction!

This feature allows you to reconstruct 3D scenes from your phone videos automatically.

### Quick Start

```bash
cd scene2sparse2dgs
bash run.sh /path/to/your/video.mp4
```

### Pipeline Steps

1. **Video Frame Extraction** (FFmpeg, 4 FPS)
2. **Smart Image Filtering** (Laplacian variance, keeps 90% best frames)
3. **COLMAP Feature Extraction** (GPU-accelerated)
4. **COLMAP Sequential Matching**
5. **COLMAP Global Mapper** (GLOMAP for robust poses)
6. **Sparse2DGS Data Preparation**
7. **Sparse2DGS Training** (30K iterations)

### Requirements

- Python 3.8+
- COLMAP 3.14+ (with CUDA support)
- FFmpeg
- 12GB+ VRAM recommended

### Documentation

See `scene2sparse2dgs/` for detailed documentation:
- `QUICKSTART.md` - Quick start guide
- `EXAMPLES.md` - Usage examples
- `ENV_SETUP.md` - Environment setup
- `PROJECT.md` - Project structure
- `GLOMAP_INTEGRATION.md` - GLOMAP integration details

---

## ⏳ Sparse View Reconstruction on DTU Dataset
### ⚽️ Running Step
* First down our preprocessed [DTU dataset](https://drive.google.com/file/d/1oaIMjZGCQhBO88ZSTKHGDPLwi2vLS_Xk/view?usp=drive_link).

```
dtu_sp
 ├── scan
 │     ├── images
 │     ├── sparse
 │     └── cam_0023.txt
 |     └── ...
 ...
```

We use the unsupervised [CLMVSNet](https://github.com/KaiqiangXiong/CL-MVSNet) to provide MVS priors. Before running, you need download the [pre-trained weights](https://drive.google.com/file/d/1y-21yh3aa18g7ORPzABR8LOOpYERmTTz/view?usp=drive_link) and set the dataset and weight paths in MVS/config.yaml.


* Set the dtu_path in scripts/train_all.py, and run the script to train the model on multiple GPUs.

```
python ./scripts/train_all.py
```

* Use the following command to render the mesh of each scene:

```
python ./scripts/render_all.py
```

### 💻 Evaluation
For evaluation, first download the [DTU ground truth](http://roboimagedata.compute.dtu.dk/?page_id=36), which includes the reference point clouds, and the [2DGS data](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), which contains scene masks and transformation matrices. Then set the corresponding paths in scripts/eval_dtu.py.


```
python ./scripts/eval_dtu.py --skip_training --skip_rendering
```


## 📰 Citation

Please cite the original paper if you use this code:
```bibtex
@article{wu2025sparse2dgs,
  title={Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views},
  author={Wu, Jiang and Li, Rui and Zhu, Yu and Guo, Rong and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2504.20378},
  year={2025}
}
```

## 📨 Acknowledgments

This fork is based on the original [Sparse2DGS](https://github.com/Wuuu3511/Sparse2DGS) by Wu et al.

The original code builds upon:
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting) - 2D Gaussian Splatting
- [CLMVSNet](https://github.com/KaiqiangXiong/CL-MVSNet) - Unsupervised MVS network

We sincerely thank the original authors for their contributions.

---

## 🤝 Contributing

This fork is actively maintained for research reproduction and practical applications. Issues and pull requests are welcome!

## 📧 Contact

For questions specific to this fork, please open a GitHub issue.

For questions about the original Sparse2DGS method, please contact the original authors.

## 📜 License

This project follows the same license as the original Sparse2DGS repository.