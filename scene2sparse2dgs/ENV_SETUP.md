# 环境配置记录

## 默认三维环境

**环境名称**: gs_linux_backup
**Python 版本**: 3.10.19
**激活命令**: `source ~/miniforge3/bin/activate gs_linux_backup`

## 环境位置

- **Conda 路径**: `/home/ltx/miniforge3`
- **环境路径**: `/home/ltx/my_envs/gs_linux_backup`
- **Python 路径**: `/home/ltx/my_envs/gs_linux_backup/bin/python`

## 已安装依赖

### 系统命令
- ✅ Python 3.10.19
- ✅ COLMAP: `/home/ltx/my_envs/gs_linux_backup/bin/colmap`
- ✅ FFmpeg: `/usr/bin/ffmpeg`
- ❌ GLOMAP: 未安装

### Python 模块
- ✅ OpenCV: 已安装
- ✅ NumPy: 已安装
- ✅ PyTorch: 已安装
- ✅ Pillow: 已安装
- ✅ PyYAML: 已安装

### GPU 信息
- ✅ CUDA available: True
- ✅ CUDA 版本: 12.8
- ✅ cuDNN 版本: 91002
- ✅ GPU 名称: NVIDIA GeForce RTX 5070
- ✅ GPU 数量: 1

## 环境检查结果

**通过率**: 12/14 (85.7%)

**缺失依赖**:
- ❌ GLOMAP（需要安装）
- ⚠️ NVIDIA Driver（WSL2 环境，预期行为）

## 使用规范

### 激活环境

**所有三维环境相关的操作必须先激活环境**:

```bash
source ~/miniforge3/bin/activate gs_linux_backup
```

### 运行 Pipeline

```bash
# 激活环境
source ~/miniforge3/bin/activate gs_linux_backup

# 进入项目目录
cd /home/ltx/projects/scene2sparse2dgs

# 检查环境
python check_env.py

# 运行 Pipeline
./run.sh video.mp4 my_scene
```

### 安装 GLOMAP

```bash
# 激活环境
source ~/miniforge3/bin/activate gs_linux_backup

# 克隆仓库
cd /tmp
git clone https://github.com/colmap/glomap.git
cd glomap

# 创建构建目录
mkdir build && cd build

# 配置编译（自动检测 GPU）
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 测试 GPU 可用性

```bash
# 激活环境
source ~/miniforge3/bin/activate gs_linux_backup

# 测试 PyTorch CUDA
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 查看完整 GPU 信息
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## 记录日期

- **配置日期**: 2026-02-15
- **最后更新**: 2026-02-15
- **测试状态**: ✅ 通过（缺少 GLOMAP）

## 注意事项

1. **必须先激活环境**: 所有三维环境相关的操作（训练、重建、渲染等）都必须先运行 `source ~/miniforge3/bin/activate gs_linux_backup`

2. **GPU 可用性**: CUDA 在 WSL2 环境中完全可用，可以正常训练

3. **GLOMAP 安装**: 需要手动编译安装，详见上面的安装说明

4. **环境隔离**: 这个环境与系统 Python 隔离，不会影响其他项目
