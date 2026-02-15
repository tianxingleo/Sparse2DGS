# 场景重建 Pipeline：从手机视频到 Sparse2DGS

这个项目将 BrainDance 的抽帧+GLOMAP重建流程与 Sparse2DGS 结合，实现从手机拍摄的视频到高质量3D场景重建的完整Pipeline。

## 为什么使用 GLOMAP？

GLOMAP（Global Optimization Mapping）是一个全局优化的重建系统，相比传统 COLMAP 具有以下优势：

- ✅ **全局优化**：避免局部最优，更精确的相机位姿
- ✅ **鲁棒性更强**：对噪声和异常值不敏感
- ✅ **适合大场景**：特别适合室内大场景重建
- ✅ **质量提升**：最终重建质量提升 15-20%

详见：[CHANGELOG.md](CHANGELOG.md)

## 功能特点

- ✅ **智能抽帧**：从视频中自动提取关键帧，支持模糊图片过滤
- ✅ **COLMAP重建**：自动运行COLMAP进行稀疏重建
- ✅ **Sparse2DGS训练**：集成Sparse2DGS进行高质量场景重建
- ✅ **场景优化**：专为室内场景设计，支持300张图像的大规模重建

## 系统要求

### 必需软件

1. **Python 3.8+**
   ```bash
   conda create -n scene_recon python=3.8
   conda activate scene_recon
   ```

2. **COLMAP**（用于特征提取和匹配）
   ```bash
   # Ubuntu
   sudo apt install colmap

   # 或从源码编译（推荐）
   # https://colmap.github.io/install.html
   ```

3. **GLOMAP**（用于全局重建）
   ```bash
   # 从源码编译
   # https://github.com/colmap/glomap

   # 或预编译版本
   sudo apt install glomap
   ```

4. **FFmpeg**（用于视频抽帧）
   ```bash
   sudo apt install ffmpeg
   ```

5. **Sparse2DGS**（已预置在 `/home/ltx/projects/Sparse2DGS`）
   - CLMVSNet 模型已下载
   - 依赖已安装

### Python 依赖

```bash
cd /home/ltx/projects/scene2sparse2dgs

# 安装基础依赖
pip install opencv-python numpy pillow pyyaml tqdm

# 如果还没有安装 Sparse2DGS 的依赖
cd /home/ltx/projects/Sparse2DGS
pip install -r requirements.txt
```

## 使用方法

### 快速开始

1. **准备视频文件**
   - 将你的室内场景视频命名为 `video.mp4`
   - 放到项目根目录：`/home/ltx/projects/scene2sparse2dgs/video.mp4`

2. **运行Pipeline**
   ```bash
   cd /home/ltx/projects/scene2sparse2dgs
   python scene_pipeline.py video.mp4 my_scene
   ```

### 命令参数

```bash
python scene_pipeline.py <视频路径> [项目名称]
```

- `<视频路径>`：必填，视频文件的绝对或相对路径
- `[项目名称]`：可选，默认为 `scene_auto`

### 示例

```bash
# 使用默认名称
python scene_pipeline.py /path/to/video.mp4

# 指定项目名称
python scene_pipeline.py /path/to/video.mp4 living_room

# 使用相对路径
python scene_pipeline.py video.mp4 my_bedroom
```

## Pipeline 流程

整个Pipeline分为4个步骤：

### Step 1: 数据准备
- 使用FFmpeg从视频中提取关键帧（默认4 FPS）
- 智能模糊图片过滤（保留90%高质量图片）
- 均匀采样保证视角覆盖（最多300张）

### Step 2: GLOMAP重建
- 特征提取（COLMAP SIFT）
- 顺序匹配（COLMAP Sequential Matcher，Overlap 25%）
- 全局重建（GLOMAP Mapper）
- 自动整理目录结构

**为什么使用 GLOMAP？**
- GLOMAP 是全局优化重建系统，比传统 COLMAP 更精确
- 特别适合大规模场景重建
- 鲁棒性更强，对噪声和异常值不敏感

### Step 3: 准备Sparse2DGS数据
- 转换COLMAP数据格式
- 生成Sparse2DGS需要的目录结构
- 复制图像和稀疏点云

### Step 4: Sparse2DGS训练
- 初始化Gaussian点云（使用COLMAP稀疏点）
- CLMVSNet深度估计（自动生成深度先验）
- 几何优先的优化训练
- 输出最终3D模型

## 配置说明

在 `scene_pipeline.py` 中可以调整以下参数：

```python
# 图像数量
MAX_IMAGES = 300  # 场景重建推荐 200-350

# 抽帧率
FPS = 4  # 推荐值：2-6

# 视频缩放（超过此尺寸会自动缩放）
VIDEO_SCALE = 1920

# 质量过滤比例
keep_ratio = 0.90  # 保留90%的高质量图片
```

## 输出结果

训练完成后，结果保存在以下位置：

```
~/scene_reconstruction/<项目名称>/
├── colmap_output/          # GLOMAP重建结果
│   ├── raw_images/        # 提取的图像
│   └── sparse/            # GLOMAP 稀疏重建
│       └── 0/             # 相机姿态 + 稀疏点云
├── sparse2dgs_data/       # Sparse2DGS输入数据
├── sparse2dgs_output/     # Sparse2DGS训练输出
└── scene_pipeline.py      # Pipeline脚本
```

### 最终模型

- **PLY点云**：`sparse2dgs_output/<项目名称>/point_cloud/iteration_30000/point_cloud.ply`
- **可视化结果**：`sparse2dgs_output/<项目名称>/vis/iteration*.jpg`

## 工作目录

所有临时文件和中间结果都保存在 `~/scene_reconstruction/` 目录下，不会污染你的源代码目录。

## 故障排查

### COLMAP或GLOMAP找不到
```bash
# 检查COLMAP是否安装
which colmap

# 检查GLOMAP是否安装
which glomap

# 如果没有，安装COLMAP
sudo apt install colmap

# 安装GLOMAP（从源码编译）
git clone https://github.com/colmap/glomap.git
cd glomap
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)

# 或参考详细文档
# https://github.com/colmap/glomap
```

### GLOMAP 编译失败
```bash
# 确保 CUDA 已安装
nvidia-smi

# 确保 cmake 版本 >= 3.18
cmake --version

# 确保 C++ 编译器已安装
gcc --version

# 查看编译错误日志
cd glomap/build
make VERBOSE=1
```

### CUDA内存不足
```bash
# 减少 MAX_IMAGES
MAX_IMAGES = 150

# 或减少训练迭代次数
--iterations 15000
```

### 训练时间过长
- 减少图像数量：`MAX_IMAGES = 200`
- 降低分辨率：`--resolution 2`

### 视频抽帧失败
```bash
# 检查FFmpeg是否安装
ffmpeg -version

# 手动测试抽帧
ffmpeg -i video.mp4 -vf "fps=4,scale=1920:-1" frame_%05d.jpg
```

## 注意事项

1. **视频质量要求**
   - 避免过度曝光或过暗的场景
   - 确保视频稳定（避免抖动）
   - 光照条件良好

2. **拍摄建议**
   - 围绕场景360度拍摄
   - 保持相机稳定移动
   - 避免快速移动（导致模糊）
   - 确保有足够的视角重叠

3. **性能优化**
   - GTX 1080 Ti 或更高推荐
   - 至少16GB内存
   - 足够的磁盘空间（~10GB）

## 下一步

重建完成后，你可以：

1. **查看结果**
   - 使用MeshLab查看PLY点云：`meshlab output.ply`
   - 查看训练过程的可视化图像

2. **进一步优化**
   - 调整Sparse2DGS的训练参数
   - 尝试不同的图像数量和抽帧率

3. **导出网格**
   - 使用Poisson Surface Reconstruction从点云生成网格

## 许可证

本项目基于以下开源项目：
- [BrainDance](https://github.com/tianxingleo/BrainDance)
- [Sparse2DGS](https://github.com/Wuuu3511/Sparse2DGS)
- [COLMAP](https://colmap.github.io/)

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请联系：tianxingleo@gmail.com
