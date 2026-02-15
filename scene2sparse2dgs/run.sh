#!/bin/bash
# 场景重建 Pipeline - 快速启动脚本

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查系统依赖..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 未安装"
        exit 1
    fi
    print_info "✓ Python 3 已安装"
    
    # 检查 COLMAP
    if ! command -v colmap &> /dev/null; then
        print_warn "COLMAP 未安装，请先安装："
        echo "  sudo apt install colmap"
        exit 1
    fi
    print_info "✓ COLMAP 已安装"
    
    # 检查 GLOMAP
    if ! command -v glomap &> /dev/null; then
        print_warn "GLOMAP 未安装，请先安装："
        echo "  编译安装: https://github.com/colmap/glomap"
        exit 1
    fi
    print_info "✓ GLOMAP 已安装"
    
    # 检查 FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        print_warn "FFmpeg 未安装，请先安装："
        echo "  sudo apt install ffmpeg"
        exit 1
    fi
    print_info "✓ FFmpeg 已安装"
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_warn "未检测到 NVIDIA 驱动，训练可能无法使用 GPU"
    else
        print_info "✓ NVIDIA CUDA 已安装"
    fi
}

# 检查 Python 依赖
check_python_dependencies() {
    print_info "检查 Python 依赖..."
    
    # 检查关键包
    python3 -c "import cv2" 2>/dev/null || print_error "opencv-python 未安装"
    python3 -c "import numpy" 2>/dev/null || print_error "numpy 未安装"
    python3 -c "import torch" 2>/dev/null || print_warn "PyTorch 未安装（Sparse2DGS 需要）"
    
    print_info "✓ Python 依赖检查完成"
}

# 显示使用帮助
show_usage() {
    cat << EOF
场景重建 Pipeline - 使用说明

用法:
    ./run.sh <视频路径> [项目名称]

参数:
    <视频路径>    视频文件的路径（必需）
    [项目名称]    项目名称（可选，默认为 scene_auto）

示例:
    ./run.sh video.mp4
    ./run.sh /path/to/video.mp4 my_scene
    ./run.sh ~/Videos/living_room.mp4 bedroom

输出:
    训练完成后，结果保存在:
    ~/scene_reconstruction/<项目名称>/sparse2dgs_output/

注意:
    - 确保视频质量良好（清晰、稳定、光照充足）
    - 推荐使用 2-6 FPS 的抽帧率（默认 4 FPS）
    - 场景重建推荐 200-350 张图像（默认 300 张）

EOF
}

# 主函数
main() {
    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════╗
║        场景重建 Pipeline v1.0                         ║
║   从手机视频到 Sparse2DGS 高质量3D重建                ║
╚═══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # 检查参数
    if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        show_usage
        exit 0
    fi
    
    VIDEO_PATH="$1"
    PROJECT_NAME="${2:-scene_auto}"
    
    # 检查视频文件
    if [ ! -f "$VIDEO_PATH" ]; then
        print_error "视频文件不存在: $VIDEO_PATH"
        exit 1
    fi
    
    print_info "视频路径: $VIDEO_PATH"
    print_info "项目名称: $PROJECT_NAME"
    
    # 检查依赖
    check_dependencies
    check_python_dependencies
    
    # 创建工作目录
    WORK_DIR="$HOME/scene_reconstruction"
    mkdir -p "$WORK_DIR"
    print_info "工作目录: $WORK_DIR"
    
    # 显示Pipeline信息
    echo ""
    print_info "Pipeline 配置:"
    echo "  - 最大图像数量: 300"
    echo "  - 抽帧率: 4 FPS"
    echo "  - 视频缩放: 1920px"
    echo "  - 训练迭代: 30000"
    echo ""
    
    print_warn "即将开始场景重建，这可能需要较长时间..."
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi
    
    # 运行Pipeline
    echo ""
    print_info "开始执行Pipeline..."
    echo ""
    
    python3 scene_pipeline.py "$VIDEO_PATH" "$PROJECT_NAME"
    
    # 检查结果
    if [ $? -eq 0 ]; then
        echo ""
        print_info "========================================="
        print_info "✓ 场景重建完成！"
        print_info "========================================="
        print_info "结果目录:"
        echo "  $WORK_DIR/$PROJECT_NAME/sparse2dgs_output/"
        print_info ""
        print_info "查看方法:"
        echo "  1. 使用 MeshLab 查看 PLY 文件:"
        echo "     meshlab $WORK_DIR/$PROJECT_NAME/sparse2dgs_output/$PROJECT_NAME/point_cloud/iteration_30000/point_cloud.ply"
        echo ""
        echo "  2. 查看训练可视化:"
        echo "     ls $WORK_DIR/$PROJECT_NAME/sparse2dgs_output/$PROJECT_NAME/vis/"
    else
        print_error "场景重建失败，请检查日志"
        exit 1
    fi
}

# 运行主函数
main "$@"
