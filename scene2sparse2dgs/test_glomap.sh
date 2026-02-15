#!/bin/bash
# 快速测试 GLOMAP 环境

set -e

echo "🔍 检查 GLOMAP 环境..."
echo ""

# 检查 COLMAP
if command -v colmap &> /dev/null; then
    echo "✅ COLMAP: $(which colmap)"
else
    echo "❌ COLMAP 未安装"
    echo "   安装方法: sudo apt install colmap"
    exit 1
fi

# 检查 GLOMAP
if command -v glomap &> /dev/null; then
    echo "✅ GLOMAP: $(which glomap)"
else
    echo "❌ GLOMAP 未安装"
    echo "   安装方法:"
    echo "   1. git clone https://github.com/colmap/glomap.git"
    echo "   2. cd glomap && mkdir build && cd build"
    echo "   3. cmake .. -DCMAKE_CUDA_ARCHITECTURES=native"
    echo "   4. make -j\$(nproc)"
    exit 1
fi

# 检查 FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg: $(which ffmpeg)"
else
    echo "❌ FFmpeg 未安装"
    echo "   安装方法: sudo apt install ffmpeg"
    exit 1
fi

# 检查 Python
if command -v python3 &> /dev/null; then
    echo "✅ Python: $(python3 --version)"
else
    echo "❌ Python 3 未安装"
    exit 1
fi

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Driver:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "⚠️  NVIDIA Driver 未检测到（可能无法使用 GPU）"
fi

echo ""
echo "✅ 所有必要组件已安装！"
echo ""
echo "下一步："
echo "1. 准备视频文件（video.mp4）"
echo "2. 运行 Pipeline: ./run.sh video.mp4 my_scene"
