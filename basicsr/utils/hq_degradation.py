import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import glob


def generate_advanced_dsec_lq(img, scale=4):
    """
    高级退化逻辑：Gamma变换 + 模糊 + 异方差噪声 + 色彩偏移
    """
    # 1. 模拟光照压缩 (非线性曝光)
    # 随机 Gamma：值越大越暗，暗部细节压缩越厉害
    gamma = np.random.uniform(2.0, 3.5)
    img_norm = img.astype(np.float32) / 255.0
    img_low = np.power(img_norm, gamma)

    # 2. 空间退化 (下采样)
    h, w = img_low.shape[:2]
    # 先轻微模糊，模拟真实镜头的点扩散函数 (PSF)
    blur_k = np.random.choice([3, 5])
    img_low = cv2.GaussianBlur(img_low, (blur_k, blur_k), 0)
    img_lq = cv2.resize(img_low, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

    # 3. 模拟传感器噪声 (混合噪声模型)
    # sigma_s: 散粒噪声系数 (信号相关)
    # sigma_r: 读取噪声系数 (静态底噪)
    sigma_s = np.random.uniform(0.01, 0.04)
    sigma_r = np.random.uniform(0.01, 0.03)

    # 计算每个像素点的噪声标准差
    noise_sigma = np.sqrt(sigma_s * img_lq + sigma_r**2)
    noise = np.random.randn(*img_lq.shape) * noise_sigma
    img_lq = img_lq + noise

    # 4. 色彩偏移 (模拟低光白平衡失真)
    color_shift = np.random.uniform(0.9, 1.1, size=(1, 1, 3))
    img_lq = img_lq * color_shift

    # 限制范围并转回 uint8
    img_lq = np.clip(img_lq * 255.0, 0, 255).astype(np.uint8)
    return img_lq


def process_dsec_sequence(gt_dir, lq_dir, scale=4):
    """
    批量处理序列文件夹
    """
    gt_path_root = Path(gt_dir)
    lq_path_root = Path(lq_dir)

    # 递归查找所有 png (支持多序列)
    img_paths = sorted(list(gt_path_root.glob("**/*.png")))

    print(f"开始生成高级退化 LQ 数据，共 {len(img_paths)} 张...")

    for img_p in tqdm(img_paths):
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        # 应用高级退化
        img_lq = generate_advanced_dsec_lq(img, scale=scale)

        # 保持原始目录结构
        rel_path = img_p.relative_to(gt_path_root)
        save_path = lq_path_root / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), img_lq)


if __name__ == "__main__":
    input_path = Path("/mnt/data1/guosr/DSEC_RetinexEVSR_test/GT")
    output_path = Path("/mnt/data1/guosr/DSEC_RetinexEVSR_test/LowLightLRx4")
    process_dsec_sequence(input_path, output_path, scale=4)
