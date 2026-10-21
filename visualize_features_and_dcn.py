import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def make_colorwheel():
    """生成 Middlebury 标准 58 色渐变色环"""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

def flow_compute_color(u, v):
    """根据 u, v 计算色环 RGB"""
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    
    rad = np.sqrt(np.square(u) + np.square(v))
    maxrad = np.percentile(rad, 99)
    if maxrad == 0:
        maxrad = 1e-5
        
    u = u / maxrad
    v = v / maxrad
    
    rad = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(-v, -u) / np.pi
    
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        
        flow_image[:, :, i] = np.floor(255 * col)
        
    return flow_image

def draw_color_wheel_legend(size=80):
    y, x = np.ogrid[-size:size, -size:size]
    mask = x**2 + y**2 <= size**2
    u = x.astype(np.float32)
    v = y.astype(np.float32)
    wheel = flow_compute_color(u, v)
    wheel[~mask] = 255 # 背景设为白色
    return wheel

def flow_to_color_with_legend(flow_tensor):
    if isinstance(flow_tensor, torch.Tensor):
        flow_np = flow_tensor.detach().cpu().numpy()
    else:
        flow_np = flow_tensor
        
    u, v = flow_np[0], flow_np[1]
    img_rgb = flow_compute_color(u, v)
    
    # 在右下角叠加色环 Legend
    legend = draw_color_wheel_legend(size=35)
    lh, lw, _ = legend.shape
    h, w, _ = img_rgb.shape
    if h > lh + 10 and w > lw + 10:
        img_rgb[h - lh - 5:h - 5, w - lw - 5:w - 5] = legend
        
    return img_rgb

def feature_to_heatmap(feature_tensor, cmap_name='viridis'):
    """将 (C, H, W) 特征图或 Mask 转换为热力图"""
    if isinstance(feature_tensor, torch.Tensor):
        feat_numpy = feature_tensor.detach().cpu().numpy()
    else:
        feat_numpy = feature_tensor
        
    # 通道维度取均值
    feat_mean = np.mean(feat_numpy, axis=0)
    feat_min, feat_max = np.min(feat_mean), np.max(feat_mean)
    
    if feat_max - feat_min == 0:
        feat_norm = np.zeros_like(feat_mean)
    else:
        feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
        
    colormap = plt.get_cmap(cmap_name)
    heatmap = (colormap(feat_norm) * 255).astype(np.uint8)[:, :, :3]
    return heatmap

def process_pt_file(pt_file_path, output_dir):
    print(f"Processing: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')
    
    base_name = os.path.splitext(os.path.basename(pt_file_path))[0]
    save_dir = os.path.join(output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)

    if 'flows' in data and 'backward' in data['flows']:
        for t, flow_t in enumerate(data['flows']['backward']):
            flow_rgb = flow_to_color_with_legend(flow_t[0])
            cv2.imwrite(os.path.join(save_dir, f'flow_backward_t{t:02d}.png'), cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))

    if 'encoder1_output' in data:
        for t, feat_t in enumerate(data['encoder1_output']):
            heatmap_rgb = feature_to_heatmap(feat_t[0], cmap_name='viridis')
            cv2.imwrite(os.path.join(save_dir, f'encoder1_feat_t{t:02d}.png'), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

    if 'decoder2_output' in data:
        for t, feat_t in enumerate(data['decoder2_output']):
            heatmap_rgb = feature_to_heatmap(feat_t[0], cmap_name='viridis')
            cv2.imwrite(os.path.join(save_dir, f'decoder2_feat_t{t:02d}.png'), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

    if 'dcn_masks' in data and data['dcn_masks'] is not None:
        for t, mask_t in enumerate(data['dcn_masks']):
            if mask_t is not None:
                mask_rgb = feature_to_heatmap(mask_t[0], cmap_name='magma')
                cv2.imwrite(os.path.join(save_dir, f'dcn_mask_t{t:02d}.png'), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    
    os.makedirs(output_folder, exist_ok=True)
    pt_files = sorted(glob(os.path.join(input_folder, "*.pt")))

    for pt_file in pt_files:
        process_pt_file(pt_file, output_folder)
    print(f"结果保存在: {output_folder}")