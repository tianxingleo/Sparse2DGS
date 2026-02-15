#!/usr/bin/env python3
"""
场景重建 Pipeline：从手机视频到 Sparse2DGS
结合 BrainDance 的抽帧+重建流程 + Sparse2DGS 训练
"""
import subprocess
import sys
import shutil
import os
import time
import datetime
from pathlib import Path
import json
import numpy as np
import cv2
import re

# ================= 配置 =================
LINUX_WORK_ROOT = Path.home() / "scene_reconstruction"
MAX_IMAGES = 300  # 场景重建需要更多视角
FPS = 4  # 抽帧率
VIDEO_SCALE = 1920  # 视频缩放

# Sparse2DGS 相关配置
SPARSE2DGS_PATH = Path("/home/ltx/projects/Sparse2DGS")
DTU_DATASET_PATH = SPARSE2DGS_PATH / "dtu_sparse"

# ================= 辅助函数 =================
def format_duration(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))

def smart_filter_blurry_images(image_folder, keep_ratio=0.90, max_images=MAX_IMAGES):
    """智能过滤模糊图片"""
    print(f"\n🧠 [智能清洗] 正在分析图片质量...")
    
    image_dir = Path(image_folder)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images:
        print("❌ 没找到图片")
        return

    trash_dir = image_dir.parent / "trash_smart"
    trash_dir.mkdir(exist_ok=True)

    img_scores = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 九宫格评分
        grid_h, grid_w = h // 3, w // 3
        max_grid_score = 0
        for r in range(3):
            for c in range(3):
                roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                score = cv2.Laplacian(roi, cv2.CV_64F).var()
                if score > max_grid_score:
                    max_grid_score = score
        
        img_scores.append((img_path, max_grid_score))
        if i % 20 == 0:
            print(f"  -> 分析中... {img_path.name}: 局部最高分 {max_grid_score:.1f}")

    # 质量清洗
    scores = [s[1] for s in img_scores]
    if not scores: return

    num_total = len(scores)
    quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
    
    print(f"\n📊 统计结果:")
    print(f"   - 图片总数: {num_total}")
    print(f"   - 质量阈值 (Bottom {(1-keep_ratio)*100:.0f}%): {quality_threshold:.2f}")

    good_images = []
    removed_count_quality = 0

    for img_path, score in img_scores:
        if score < quality_threshold:
            shutil.move(str(img_path), str(trash_dir / img_path.name))
            removed_count_quality += 1
        else:
            good_images.append(img_path)

    print(f"   -> 第一轮清洗完成: 剔除 {removed_count_quality} 张废片，剩余 {len(good_images)} 张。")

    # 数量控制
    removed_count_quantity = 0
    
    if len(good_images) > max_images:
        print(f"   ⚠️ 合格图片 ({len(good_images)}) 仍超过上限 ({max_images})")
        print(f"   -> 执行【均匀采样】以保证视角覆盖...")
        
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
                removed_count_quantity += 1
    else:
        print(f"   ✅ 合格图片数量 ({len(good_images)}) 未超标，全部保留。")

    total_removed = removed_count_quality + removed_count_quantity
    final_count = num_total - total_removed
    print(f"✨ 清洗结束: 共移除 {total_removed} 张，最终保留 {final_count} 张。")

def run_command(cmd, description, env=None, cwd=None):
    """运行命令并输出日志"""
    print(f"\n⚡ {description}...")
    try:
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            env=env or os.environ.copy(),
            bufsize=1,
            cwd=cwd
        ) as process:
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
    except Exception as e:
        print(f"❌ {description} 失败: {e}")
        raise e

def prepare_sparse2dgs_data(colmap_output, target_dir, scene_name):
    """
    准备 Sparse2DGS 数据格式
    参考 Sparse2DGS 的数据结构
    """
    print(f"\n📦 [数据转换] 准备 Sparse2DGS 数据...")
    
    target_dir = Path(target_dir)
    scene_dir = target_dir / scene_name
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"
    
    # 创建目录结构
    scene_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图像
    colmap_images = colmap_output / "raw_images"
    if not colmap_images.exists():
        colmap_images = colmap_output / "images"
    
    image_count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        for img_path in colmap_images.glob(ext):
            shutil.copy2(str(img_path), str(images_dir / img_path.name))
            image_count += 1
    
    print(f"    ✅ 已复制 {image_count} 张图像")
    
    # 复制 COLMAP sparse 数据
    colmap_sparse = colmap_output / "colmap" / "sparse" / "0"
    if not colmap_sparse.exists():
        colmap_sparse = colmap_output / "colmap" / "sparse"
    
    sparse_files_found = False
    if colmap_sparse.exists():
        for file in colmap_sparse.glob("*"):
            if file.suffix in ['.bin', '.txt']:
                shutil.copy2(str(file), str(sparse_dir / file.name))
                sparse_files_found = True
    
    if not sparse_files_found:
        print("❌ 未找到 COLMAP sparse 数据")
        return None
    
    print(f"    ✅ Sparse2DGS 数据已准备: {scene_dir}")
    return scene_dir

def run_sparse2dgs_training(scene_dir, output_dir, scan_name):
    """
    运行 Sparse2DGS 训练
    """
    print(f"\n🚀 [Sparse2DGS] 开始训练...")
    
    # 训练脚本路径
    train_script = SPARSE2DGS_PATH / "train.py"
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    args = [
        "python", str(train_script),
        "--source_path", str(scene_dir),
        "--model_path", str(output_dir / scan_name),
        "--images", "images",
        "--eval",
        "--iterations", "30000",
        "--resolution", "1",
        "--white_background",
    ]
    
    # 运行训练
    run_command(args, "训练 Sparse2DGS", cwd=str(SPARSE2DGS_PATH))
    
    print(f"\n✅ Sparse2DGS 训练完成！")
    print(f"   输出目录: {output_dir / scan_name}")
    
    return output_dir / scan_name

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    """完整的场景重建 Pipeline"""
    
    global_start_time = time.time()
    print(f"\n🚀 [场景重建 Pipeline] 启动任务: {project_name}")
    print(f"🕒 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    colmap_output_dir = work_dir / "colmap_output"
    sparse2dgs_data_dir = work_dir / "sparse2dgs_data"
    sparse2dgs_output_dir = work_dir / "sparse2dgs_output"
    
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    
    # ================= Step 1: 数据准备 =================
    step1_start = time.time()
    
    print(f"\n🎥 [Step 1/4] 数据准备")
    
    # 创建工作目录
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制视频
    shutil.copy(str(video_src), str(work_dir / video_src.name))
    
    # 创建临时目录
    temp_dir = work_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_images_dir = colmap_output_dir / "raw_images"
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg 抽帧
    print(f"    -> 正在抽帧...")
    cap = cv2.VideoCapture(str(work_dir / video_src.name))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    vf_param = f"fps={FPS}"
    if width > VIDEO_SCALE:
        vf_param = f"scale={VIDEO_SCALE}:-1,fps={FPS}"
    
    try:
        run_command([
            "ffmpeg", "-y", "-i", str(work_dir / video_src.name),
            "-vf", vf_param, "-q:v", "2",
            str(temp_dir / "frame_%05d.jpg")
        ], "抽帧")
    except Exception as e:
        print(f"⚠️ FFmpeg 抽帧结束: {e}")
    
    # 智能过滤
    smart_filter_blurry_images(temp_dir, keep_ratio=0.90, max_images=MAX_IMAGES)
    
    # 迁移图片
    print(f"    -> 正在迁移图片...")
    all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
    total_candidates = len(all_candidates)
    
    final_images_list = []
    if total_candidates > MAX_IMAGES:
        print(f"    ⚠️ 图片过多 ({total_candidates}), 正在均匀选取 {MAX_IMAGES} 张...")
        indices = np.linspace(0, total_candidates - 1, MAX_IMAGES, dtype=int)
        indices = sorted(list(set(indices)))
        
        for idx in indices:
            final_images_list.append(all_candidates[idx])
    else:
        final_images_list = all_candidates
    
    for img_path in final_images_list:
        shutil.copy2(str(img_path), str(extracted_images_dir / img_path.name))
    
    print(f"    ✅ 已迁移 {len(final_images_list)} 张图片")
    shutil.rmtree(temp_dir)
    
    step1_duration = time.time() - step1_start
    print(f"⏱️ [Step 1 完成] 耗时: {format_duration(step1_duration)}")
    
    # ================= Step 2: COLMAP Global Mapper 重建 =================
    step2_start = time.time()
    
    print(f"\n🗺️  [Step 2/4] COLMAP Global Mapper 重建 (GLOMAP 功能）")
    
    # 查找 COLMAP（用于特征提取和匹配）
    system_colmap_exe = shutil.which("colmap")
    if not system_colmap_exe:
        if os.path.exists("/usr/local/bin/colmap"):
            system_colmap_exe = "/usr/local/bin/colmap"
        else:
            raise FileNotFoundError("❌ 无法找到 colmap 可执行文件")
    
    # 查找 COLMAP（用于全局重建）
    system_colmap_exe = shutil.which("colmap")
    if not system_colmap_exe:
        if os.path.exists("/usr/local/bin/colmap"):
            system_colmap_exe = "/usr/local/bin/colmap"
        else:
            raise FileNotFoundError("❌ 无法找到 colmap 可执行文件")
    
    print(f"🎯 COLMAP (包含 GLOMAP global_mapper): {system_colmap_exe}")
    
    database_path = colmap_output_dir / "database.db"
    
    # 特征提取（使用 COLMAP）
    run_command([
        system_colmap_exe, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1"
    ], "特征提取 (COLMAP)")
    
    # 顺序匹配（使用 COLMAP）
    run_command([
        system_colmap_exe, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "25"
    ], "顺序匹配 (COLMAP)")
    
    # COLMAP Global Mapper（GLOMAP 全局重建）
    global_mapper_output_dir = colmap_output_dir / "sparse"
    global_mapper_output_dir.mkdir(parents=True, exist_ok=True)
    
    run_command([
        system_colmap_exe, "global_mapper",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--output_path", str(global_mapper_output_dir)
    ], "COLMAP Global Mapper (GLOMAP 全局重建)")
    
    # 整理目录结构（GLOMAP 输出可能在 sparse/0 或 sparse 根目录）
    colmap_sparse_root = colmap_output_dir / "sparse"
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    
    # 检查 GLOMAP 输出位置
    sparse_output = None
    possible_locations = [
        colmap_output_dir / "sparse" / "0",  # 标准位置
        colmap_output_dir / "sparse",         # 根目录
    ]
    
    for loc in possible_locations:
        if all((loc / f).exists() for f in required_files):
            sparse_output = loc
            break
    
    if sparse_output is None:
        # 查找模型文件
        for root, dirs, files in os.walk(colmap_sparse_root):
            if all(f in files for f in required_files):
                src_path = Path(root)
                if not (colmap_output_dir / "sparse" / "0").exists():
                    (colmap_output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
                for f in required_files:
                    shutil.move(str(src_path / f), str(colmap_output_dir / "sparse" / "0" / f))
                sparse_output = colmap_output_dir / "sparse" / "0"
                break
    
    if sparse_output is None:
        raise RuntimeError("GLOMAP 重建失败，未找到输出文件")
    
    print(f"✅ COLMAP Global Mapper 重建完成（GLOMAP 功能）")
    
    step2_duration = time.time() - step2_start
    print(f"⏱️ [Step 2 完成] 耗时: {format_duration(step2_duration)}")
    print(f"   输出: {sparse_output}")
    
    # ================= Step 3: 准备 Sparse2DGS 数据 =================
    step3_start = time.time()
    
    print(f"\n📦 [Step 3/4] 准备 Sparse2DGS 数据")
    
    scene_dir = prepare_sparse2dgs_data(colmap_output_dir, sparse2dgs_data_dir, project_name)
    
    if scene_dir is None:
        raise RuntimeError("数据准备失败")
    
    step3_duration = time.time() - step3_start
    print(f"⏱️ [Step 3 完成] 耗时: {format_duration(step3_duration)}")
    
    # ================= Step 4: Sparse2DGS 训练 =================
    step4_start = time.time()
    
    print(f"\n🚀 [Step 4/4] Sparse2DGS 训练")
    
    output_dir = run_sparse2dgs_training(
        scene_dir,
        sparse2dgs_output_dir,
        project_name
    )
    
    step4_duration = time.time() - step4_start
    print(f"⏱️ [Step 4 完成] 耗时: {format_duration(step4_duration)}")
    
    # ================= 完成 =================
    total_time = time.time() - global_start_time
    
    print(f"\n✅ =============================================")
    print(f"🎉 场景重建完成！")
    print(f"📂 最终输出: {output_dir}")
    print(f"⏱️ 总耗时: {format_duration(total_time)}")
    print(f"✅ =============================================")
    
    # 回传结果
    if output_dir and output_dir.exists():
        target_dir = Path(__file__).parent / "results"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file in output_dir.glob("*.ply"):
            shutil.copy2(str(file), str(target_dir / file.name))
            print(f"📦 已复制: {file.name}")
        
        print(f"\n📂 结果已保存到: {target_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scene_pipeline.py <视频路径> [项目名称]")
        print("示例: python scene_pipeline.py video.mp4 my_scene")
        sys.exit(1)
    
    video_file = Path(sys.argv[1])
    project_name = sys.argv[2] if len(sys.argv) > 2 else "scene_auto"
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)
    
    run_pipeline(video_file, project_name)
