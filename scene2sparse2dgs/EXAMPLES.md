# 使用示例

## 示例 1: 快速开始

### 步骤 1: 检查环境

```bash
cd /home/ltx/projects/scene2sparse2dgs
python check_env.py
```

**预期输出：**
```
============================================================
场景重建 Pipeline - 环境检查
============================================================

【系统命令】
✓ Python 3: /usr/bin/python3
✓ COLMAP: /usr/local/bin/colmap
✓ FFmpeg: /usr/bin/ffmpeg
✓ NVIDIA Driver: /usr/bin/nvidia-smi

【Python 模块】
✓ OpenCV: 已安装
✓ NumPy: 已安装
✓ PyTorch: 已安装
✓ Pillow: 已安装
✓ PyYAML: 已安装

【文件和目录】
✓ Sparse2DGS 目录: /home/ltx/projects/Sparse2DGS
✓ CLMVSNet 模型: /home/ltx/projects/Sparse2DGS/model_clmvsnet.ckpt
✓ MVS 配置文件: /home/ltx/projects/Sparse2DGS/MVS/config.yaml
✓ Pipeline 脚本: /home/ltx/projects/scene2sparse2dgs/scene_pipeline.py

【GPU 信息】
✓ GPU 信息: NVIDIA GeForce RTX 3090, 24384 MiB
✓ CUDA 版本: 11.7
✓ cuDNN 版本: 8500

============================================================
检查结果: 13/13 项通过
✓ 所有检查通过！可以开始使用 Pipeline。
```

### 步骤 2: 准备视频

```bash
# 将你的视频复制到项目目录
cp ~/Videos/living_room.mp4 /home/ltx/projects/scene2sparse2dgs/video.mp4
```

### 步骤 3: 运行 Pipeline

```bash
# 使用启动脚本（推荐）
./run.sh video.mp4 living_room

# 或使用 Python 脚本
python scene_pipeline.py video.mp4 living_room
```

**预期输出：**
```
🚀 [场景重建 Pipeline] 启动任务: living_room
🕒 开始时间: 2026-02-15 19:30:00

🎥 [Step 1/4] 数据准备
    -> 正在抽帧...
⚡ 抽帧...
[ffmpeg 日志...]

🧠 [智能清洗] 正在分析图片质量...
📊 统计结果:
   - 图片总数: 285
   - 质量阈值 (Bottom 10%): 45.23
✨ 清洗结束: 共移除 40 张，最终保留 245 张。

⏱️ [Step 1 完成] 耗时: 00:02:30

🗺️  [Step 2/4] COLMAP 重建
🎯 COLMAP: /usr/local/bin/colmap
⚡ 特征提取...
[COLMAP 日志...]

⚡ 顺序匹配...
[COLMAP 日志...]

⚡ 稀疏重建...
[COLMAP 日志...]

✅ COLMAP 重建完成
⏱️ [Step 2 完成] 耗时: 00:15:20

📦 [Step 3/4] 准备 Sparse2DGS 数据
    ✅ 已复制 245 张图像
    ✅ Sparse2DGS 数据已准备: ~/scene_reconstruction/living_room/sparse2dgs_data/living_room
⏱️ [Step 3 完成] 耗时: 00:00:45

🚀 [Step 4/4] Sparse2DGS 训练
⚡ 训练 Sparse2DGS...
[Sparse2DGS 训练日志...]

✅ Sparse2DGS 训练完成！
   输出目录: ~/scene_reconstruction/living_room/sparse2dgs_output/living_room
⏱️ [Step 4 完成] 耗时: 00:45:30

✅ =============================================
🎉 场景重建完成！
📂 最终输出: ~/scene_reconstruction/living_room/sparse2dgs_output/living_room
⏱️ 总耗时: 01:04:05
✅ =============================================

📦 已复制: point_cloud.ply

📂 结果已保存到: /home/ltx/projects/scene2sparse2dgs/results
```

## 示例 2: 自定义参数

### 场景：小房间（建议减少图像）

```bash
# 修改 scene_pipeline.py 中的参数
# MAX_IMAGES = 150  # 减少图像数量

python scene_pipeline.py small_room.mp4 small_room
```

### 场景：大空间（建议增加图像）

```bash
# 修改 scene_pipeline.py 中的参数
# MAX_IMAGES = 400  # 增加图像数量
# FPS = 3  # 降低抽帧率，获取更多帧

python scene_pipeline.py large_hall.mp4 large_hall
```

### 场景：高精度重建（增加训练迭代）

```bash
# 修改 scene_pipeline.py 中的 run_sparse2dgs_training 函数
# "--iterations", "50000",  # 增加迭代次数

python scene_pipeline.py living_room.mp4 living_room_high_quality
```

## 示例 3: 批量处理

处理多个视频：

```bash
#!/bin/bash
# batch_process.sh

videos=(
    "living_room.mp4"
    "bedroom.mp4"
    "kitchen.mp4"
)

for video in "${videos[@]}"; do
    project_name="${video%.mp4}"
    echo "处理: $video -> $project_name"
    python scene_pipeline.py "$video" "$project_name"
done
```

运行：
```bash
chmod +x batch_process.sh
./batch_process.sh
```

## 示例 4: 查看结果

### 使用 MeshLab 查看 PLY 文件

```bash
# 安装 MeshLab
sudo apt install meshlab

# 查看结果
meshlab ~/scene_reconstruction/living_room/sparse2dgs_output/living_room/point_cloud/iteration_30000/point_cloud.ply
```

### 查看训练可视化

```bash
# 列出所有可视化图像
ls ~/scene_reconstruction/living_room/sparse2dgs_output/living_room/vis/

# 使用图像查看器打开
eog ~/scene_reconstruction/living_room/sparse2dgs_output/living_room/vis/iteration_30000.jpg
```

### 导出为其他格式

```bash
# 导出为 OBJ
meshlabserver -i point_cloud.ply -o output.obj

# 导出为 STL
meshlabserver -i point_cloud.ply -o output.stl
```

## 示例 5: 错误处理

### COLMAP 匹配率过低

**症状：**
```
❌ COLMAP only found poses for 25.00% of images. This is low.
```

**解决方案：**
```bash
# 1. 检查视频质量
# 确保视频清晰、稳定、光照充足

# 2. 调整抽帧参数
# 修改 scene_pipeline.py
# FPS = 6  # 增加抽帧率，获取更多图像
# keep_ratio = 0.95  # 保留更多图像

# 3. 重新运行
python scene_pipeline.py video.mp4 scene_v2
```

### CUDA 内存不足

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 1. 减少图像数量
# 修改 scene_pipeline.py
MAX_IMAGES = 150

# 2. 降低分辨率
# 在 run_sparse2dgs_training 函数中修改
"--resolution", "2",  # 降低分辨率

# 3. 减少批量大小（修改 Sparse2DGS 源码）
# 在 Sparse2DGS/train.py 中调整 batch_size
```

### MVS 深度估计失败

**症状：**
```
❌ MVS 深度估计失败
```

**解决方案：**
```bash
# 1. 检查 CLMVSNet 模型
ls -lh /home/ltx/projects/Sparse2DGS/model_clmvsnet.ckpt

# 2. 检查配置文件
cat /home/ltx/projects/Sparse2DGS/MVS/config.yaml

# 3. 确保 Sparse2DGS 依赖已安装
cd /home/ltx/projects/Sparse2DGS
pip install -r requirements.txt
```

## 示例 6: 性能优化

### 使用多 GPU

```bash
# 修改 scene_pipeline.py 中的训练命令
# 添加 CUDA_VISIBLE_DEVICES
env["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用 GPU 0 和 1

python scene_pipeline.py video.mp4 multi_gpu_scene
```

### 减少训练时间

```bash
# 1. 减少迭代次数
# 修改 scene_pipeline.py
"--iterations", "15000",  # 减少到 15000

# 2. 降低分辨率
"--resolution", "2",  # 降低分辨率

# 3. 减少图像数量
MAX_IMAGES = 150
```

### 提高重建质量

```bash
# 1. 增加迭代次数
"--iterations", "50000",  # 增加到 50000

# 2. 使用更高分辨率
"--resolution", "1",  # 原始分辨率

# 3. 增加图像数量
MAX_IMAGES = 400

# 4. 使用更好的抽帧策略
FPS = 6  # 获取更多帧
keep_ratio = 0.95  # 保留更多图像
```

## 示例 7: 与其他工具集成

### 集成到 Blender

```python
# 在 Blender 中导入 PLY
import bpy

# 导入点云
bpy.ops.import_mesh.ply(filepath="point_cloud.ply")

# 应用 Poisson Surface Reconstruction
bpy.ops.object.duplicate()
bpy.ops.object.modifier_add(type='SKIN')
bpy.ops.object.skin_resize()
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.skin_loose()
bpy.ops.mesh.skin_mark_loose()
bpy.ops.object.mode_set(mode='OBJECT')
```

### 集成到 Unreal Engine

```python
# 使用 Datasmooth 导入
# 1. 转换 PLY 为 FBX
meshlabserver -i point_cloud.ply -o output.fbx

# 2. 在 Unreal Engine 中导入
# File > Import Unreal Datasmith
```

### 集成到 Unity

```csharp
// 在 Unity 中导入 PLY
using UnityEngine;

public class ImportPLY : MonoBehaviour
{
    public string plyFile;
    
    void Start()
    {
        // 使用插件导入 PLY
        // 例如: UnityPLYImporter
    }
}
```

## 总结

通过这些示例，你应该能够：

1. ✅ 快速开始使用 Pipeline
2. ✅ 自定义参数适应不同场景
3. ✅ 批量处理多个视频
4. ✅ 查看和导出结果
5. ✅ 处理常见错误
6. ✅ 优化性能和质量
7. ✅ 集成到其他工具

**祝你重建成功！🎉**
