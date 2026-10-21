import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_dsec_lq(gt_dir, lq_dir, scale=4, light_factor=0.5):
    """
    Prepare LQ images for DSEC training and evaluation.

    gt_dir: 原图路径
    lq_dir: 输出路径
    scale: 缩放倍数 (4代表 1/4)
    light_factor: 光强系数 (0.3-0.5，根据需求调整)
    """
    os.makedirs(lq_dir, exist_ok=True)
    gt_paths = sorted(list(Path(gt_dir).glob('*.png')))

    if not gt_paths:
        print(f"no image found in {gt_dir} ")
        return

    print(f"共 {len(gt_paths)} 张图片...")

    for path in tqdm(gt_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"错误图片: {path}")
            continue

        h, w = img.shape[:2]
        lq_w, lq_h = w // scale, h // scale

        img_lq = cv2.resize(img, (lq_w, lq_h), interpolation=cv2.INTER_AREA)
        img_lq = img_lq.astype(np.float32) * light_factor
        img_lq = np.clip(img_lq, 0, 255).astype(np.uint8)

        save_path = Path(lq_dir) / path.name
        cv2.imwrite(str(save_path), img_lq)


if __name__ == '__main__':
    input_path = '/root/autodl-tmp/DSEC_RetinexEVSR/GT/zurich_city_01_a'
    output_path = '/root/autodl-tmp/DSEC_RetinexEVSR/LowLightLRx4/zurich_city_01_a'
    prepare_dsec_lq(input_path, output_path, scale=4, light_factor=1)