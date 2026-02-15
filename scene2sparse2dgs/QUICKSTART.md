# 快速开始指南

## 1. 安装依赖

### 系统依赖
```bash
# 安装 COLMAP
sudo apt install colmap

# 安装 GLOMAP（从源码编译）
# https://github.com/colmap/glomap
git clone https://github.com/colmap/glomap.git
cd glomap
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j

# 安装 FFmpeg
sudo apt install ffmpeg

# 检查 NVIDIA 驱动
nvidia-smi
```

### Python 依赖
```bash
cd /home/ltx/projects/scene2sparse2dgs

# 安装基础依赖
pip install -r requirements.txt

# 安装 Sparse2DGS 依赖
cd ../Sparse2DGS
pip install -r requirements.txt
```

## 2. 准备视频

将你的室内场景视频放到项目目录：
```bash
cp /path/to/your/video.mp4 /home/ltx/projects/scene2sparse2dgs/video.mp4
```

## 3. 运行 Pipeline

### 使用启动脚本（推荐）
```bash
cd /home/ltx/projects/scene2sparse2dgs
./run.sh video.mp4 my_scene
```

### 使用 Python 脚本
```bash
python scene_pipeline.py video.mp4 my_scene
```

## 4. 查看结果

训练完成后，结果保存在：
```
~/scene_reconstruction/my_scene/sparse2dgs_output/
```

使用 MeshLab 查看：
```bash
meshlab ~/scene_reconstruction/my_scene/sparse2dgs_output/my_scene/point_cloud/iteration_30000/point_cloud.ply
```

## 5. 常见问题

### Q: 训练需要多长时间？
A: 取决于硬件配置：
- RTX 3090: 约 30-60 分钟
- RTX 2080 Ti: 约 1-2 小时
- GTX 1080 Ti: 约 2-3 小时

### Q: 内存不足怎么办？
A: 在 `scene_pipeline.py` 中修改：
```python
MAX_IMAGES = 150  # 减少图像数量
```

### Q: 如何提高重建质量？
A:
1. 增加图像数量：`MAX_IMAGES = 350`
2. 增加训练迭代：`--iterations 50000`
3. 确保视频质量良好

### Q: 能否重建室外场景？
A: 可以，但需要调整参数：
```python
--white_background  # 移除此参数
```

## 6. 拍摄建议

为了获得最佳的重建效果：

✅ **推荐做法**
- 围绕场景360度拍摄
- 保持相机稳定移动
- 避免快速移动
- 确保光照充足
- 保证视角重叠

❌ **避免**
- 手持抖动
- 快速平移
- 过度曝光
- 拍摄移动物体

## 7. 进阶使用

### 自定义参数
编辑 `scene_pipeline.py`：
```python
MAX_IMAGES = 300       # 图像数量
FPS = 4                # 抽帧率
VIDEO_SCALE = 1920     # 视频缩放
```

### 调整训练参数
在 `run_sparse2dgs_training()` 函数中修改：
```python
"--iterations", "50000",    # 增加迭代次数
"--resolution", "1",         # 高分辨率
```

## 8. 输出文件说明

```
sparse2dgs_output/<项目名称>/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply          # 最终点云模型
├── vis/
│   ├── iteration_7000.jpg            # 训练过程可视化
│   ├── iteration_15000.jpg
│   └── iteration_30000.jpg
├── cfg_args                          # 配置文件
└── cam_properties.json              # 相机参数
```

## 9. 下一步

训练完成后，你可以：

1. **生成网格**
   ```bash
   # 使用 Poisson Surface Reconstruction
   pcl_viewer point_cloud.ply
   # 或使用 MeshLab 的 "Filters > Reconstruction > Surface Reconstruction: Screened Poisson"
   ```

2. **纹理映射**
   ```bash
   # 使用 MeshLab 添加纹理
   Filters > Texture > Parametrization and texturing
   ```

3. **导出到其他格式**
   - OBJ: MeshLab 导出
   - GLTF: 使用 Blender 转换
   - USD: 使用 Unreal Engine 导入

## 10. 获取帮助

遇到问题？

1. 查看 [README.md](README.md) 了解详细信息
2. 检查日志文件：`~/scene_reconstruction/<项目名称>/`
3. 提交 Issue 到 GitHub

---

祝你重建成功！🎉
