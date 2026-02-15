#!/usr/bin/env python3
"""
环境检查脚本 - 验证所有依赖是否正确安装
"""
import subprocess
import sys
from pathlib import Path

# 颜色输出
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_colored(message, color):
    print(f"{color}{message}{NC}")

def check_command(cmd, name):
    """检查命令是否存在"""
    try:
        result = subprocess.run(['which', cmd], capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            print_colored(f"✓ {name}: {path}", GREEN)
            return True
        else:
            print_colored(f"✗ {name}: 未安装", RED)
            return False
    except Exception as e:
        print_colored(f"✗ {name}: 检查失败 - {e}", RED)
        return False

def check_python_module(module, name):
    """检查 Python 模块是否可导入"""
    try:
        __import__(module)
        print_colored(f"✓ {name}: 已安装", GREEN)
        return True
    except ImportError:
        print_colored(f"✗ {name}: 未安装", RED)
        return False

def check_file(path, name):
    """检查文件是否存在"""
    file_path = Path(path)
    if file_path.exists():
        print_colored(f"✓ {name}: {path}", GREEN)
        return True
    else:
        print_colored(f"✗ {name}: 未找到 - {path}", RED)
        return False

def main():
    print_colored("=" * 60, BLUE)
    print_colored("场景重建 Pipeline - 环境检查", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    
    # 检查系统命令
    print_colored("【系统命令】", BLUE)
    commands = [
        ('python3', 'Python 3'),
        ('colmap', 'COLMAP'),
        ('glomap', 'GLOMAP'),
        ('ffmpeg', 'FFmpeg'),
        ('nvidia-smi', 'NVIDIA Driver'),
    ]
    
    cmd_results = []
    for cmd, name in commands:
        result = check_command(cmd, name)
        cmd_results.append(result)
    
    print()
    
    # 检查 Python 模块
    print_colored("【Python 模块】", BLUE)
    modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
    ]
    
    module_results = []
    for module, name in modules:
        result = check_python_module(module, name)
        module_results.append(result)
    
    print()
    
    # 检查文件和目录
    print_colored("【文件和目录】", BLUE)
    files = [
        ('/home/ltx/projects/Sparse2DGS', 'Sparse2DGS 目录'),
        ('/home/ltx/projects/Sparse2DGS/model_clmvsnet.ckpt', 'CLMVSNet 模型'),
        ('/home/ltx/projects/Sparse2DGS/MVS/config.yaml', 'MVS 配置文件'),
        ('/home/ltx/projects/scene2sparse2dgs/scene_pipeline.py', 'Pipeline 脚本'),
    ]
    
    file_results = []
    for path, name in files:
        result = check_file(path, name)
        file_results.append(result)
    
    print()
    
    # 检查 GPU 信息
    print_colored("【GPU 信息】", BLUE)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_colored(f"✓ GPU 信息: {result.stdout.strip()}", GREEN)
        else:
            print_colored("⚠ GPU 检测失败", YELLOW)
    except:
        print_colored("⚠ 无法获取 GPU 信息", YELLOW)
    
    # 检查 CUDA 版本
    try:
        import torch
        if torch.cuda.is_available():
            print_colored(f"✓ CUDA 版本: {torch.version.cuda}", GREEN)
            print_colored(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}", GREEN)
        else:
            print_colored("⚠ CUDA 不可用（PyTorch 无法检测到 GPU）", YELLOW)
    except:
        print_colored("⚠ 无法检查 CUDA 版本", YELLOW)
    
    print()
    print_colored("=" * 60, BLUE)
    
    # 统计结果
    total_checks = len(cmd_results) + len(module_results) + len(file_results)
    passed_checks = sum(cmd_results) + sum(module_results) + sum(file_results)
    
    print_colored(f"检查结果: {passed_checks}/{total_checks} 项通过", 
                  GREEN if passed_checks == total_checks else YELLOW)
    
    if passed_checks == total_checks:
        print_colored("✓ 所有检查通过！可以开始使用 Pipeline。", GREEN)
        return 0
    else:
        print_colored("✗ 部分检查失败，请先解决依赖问题。", RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())
