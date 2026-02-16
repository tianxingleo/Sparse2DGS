from .convert_gs_to_mvs import *
from .get_mvs_model import get_mvs_model
import torch
import torch.nn.functional as F
import cv2
def get_mvs_depth(view_cam, scan=None):
    mvs_model = get_mvs_model().eval()
    view_cam = convert_dataset_dtu(view_cam, scan=scan)
    
    # 提前准备所有投影矩阵（CPU）
    proj_mat_list = [view.proj_mat for view in view_cam]
    
    for i in range(len(view_cam)):
        print(f'cal {view_cam[i].image_name} mvs depth (VRAM optimization for 12GB)')
        
        # --- 显存优化 1: 视图选择 (Windowed Source View Selection) ---
        # 12GB 显存建议只选 4-6 个源视图。
        num_source_views = 4  
        indices = []
        # 尝试选择前后的视角
        for offset in range(1, num_source_views + 1):
            if i + offset < len(view_cam): indices.append(i + offset)
            if i - offset >= 0: indices.append(i - offset)
            if len(indices) >= num_source_views: break
        
        # 始终将参考帧放在第 0 位
        final_indices = [i] + indices[:num_source_views]
        
        # --- 显存优化 2: 懒加载图片到 GPU ---
        image_list_mini = []
        for idx in final_indices:
            # 这里的 original_image 已经在之前的内存中，但我们确保只在这个循环内占用显存空间
            img = view_cam[idx].original_image.cuda()
            image_list_mini.append(img)
            
        images = torch.stack(image_list_mini)[None]  # [1, 5, 3, H, W]
        
        # 整理投影矩阵
        mini_proj_list = [proj_mat_list[idx].clone().cuda() for idx in final_indices]
        
        # --- 显存优化 3: 分辨率缩减与对齐 ---
        max_dim = 800 # 进一步降低分辨率以应对可能的 RAM/VRAM 双重压力
        curr_h, curr_w = images.shape[3], images.shape[4]
        
        target_h, target_w = curr_h, curr_w
        if curr_h > max_dim or curr_w > max_dim:
            scale = max_dim / max(curr_h, curr_w)
            target_h, target_w = int(curr_h * scale), int(curr_w * scale)
        
        # 必须是 32 的倍数以满足 CLMVSNet 的 UNet 结构
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32
        
        if target_h != curr_h or target_w != curr_w:
            scale_h = target_h / curr_h
            scale_w = target_w / curr_w
            images = F.interpolate(images.flatten(0, 1), size=(target_h, target_w), mode='bilinear', align_corners=True).unflatten(0, (1, -1))
            
            # 重要：同步更新投影矩阵中的内参 (Intrinsic)
            for pj in mini_proj_list:
                pj[1, 0, 0] *= scale_w # fx
                pj[1, 1, 1] *= scale_h # fy
                pj[1, 0, 2] *= scale_w # cx
                pj[1, 1, 2] *= scale_h # cy

        proj_matrices = torch.stack(mini_proj_list) # [N, 2, 4, 4]
        
        # 此时才将 Stage 矩阵放到 GPU
        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
        stage1_pjmats = proj_matrices.clone()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage1_pjmats[None],
            "stage2": stage2_pjmats[None],
            "stage3": proj_matrices[None]
        }

        data = {
            "proj_matrices": proj_matrices_ms,
            "imgs": images,
            "init_depth_hypotheses": view_cam[i].init_depth_hypotheses[None].cuda()
        }
        
        with torch.no_grad():
            mvs_out_dict = mvs_model(data, mode='test')
            
        depth = mvs_out_dict["depth"]
        photometric_confidence = mvs_out_dict["photometric_confidence"]
        h_orig, w_orig = view_cam[i].original_image.shape[1:]
        
        # 将深度图恢复到原始分辨率
        if depth.shape[2:] != (h_orig, w_orig):
            depth = F.interpolate(depth[None], size=(h_orig, w_orig), mode='bilinear', align_corners=True)[0]
            photometric_confidence = F.interpolate(photometric_confidence[None], size=(h_orig, w_orig), mode='bilinear', align_corners=True)[0]

        point = depth2wpoint(depth, K=view_cam[i].K.float(), w2c=view_cam[i].w2c.float(), w=w_orig, h=h_orig)
        
        setattr(view_cam[i], 'depth', depth)
        setattr(view_cam[i], 'photometric_confidence', photometric_confidence)
        setattr(view_cam[i], 'points', point.cpu()) # 移回 CPU，大幅减少 GPU 显存占用
        
        # 优化 feature 存储：将特征图插值回原始分辨率（720P），以匹配点云数量
        ref_fea = mvs_out_dict["ref_fea"] # (B, C, H_mvs, W_mvs)
        if ref_fea.shape[2:] != (h_orig, w_orig):
            ref_fea = F.interpolate(ref_fea, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
            
        ref_fea = ref_fea.cpu() # 保持 (B, C, H, W) 维度，以兼容 train.py 中的 [0] 索引
        if ref_fea.dtype == torch.float32:
            ref_fea = ref_fea.half() # 使用半精度存储在 RAM 中
        
        setattr(view_cam[i], 'feature', ref_fea)
        
        # --- 显存优化 4: 强制清理 ---
        del data, mvs_out_dict, images, image_list_mini, proj_matrices_ms, point, depth, photometric_confidence, ref_fea
        torch.cuda.empty_cache()
        
    return view_cam
