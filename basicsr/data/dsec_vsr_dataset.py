import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import torch
import cv2
import h5py
import random
import torch.nn.functional as F

from os import path as osp
from torch.utils import data as data
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor


@DATASET_REGISTRY.register()
class DSECVSRDataset(data.Dataset):
    def __init__(self, opt):
        super(DSECVSRDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt["dataroot_gt"]
        self.lq_root = opt["dataroot_lq"]
        self.voxel_root = opt["dataroot_voxel"]
        self.scale = opt.get("scale", 4)
        self.phase = opt.get("phase", "test")
        self.num_bins = 15
        self.num_frame = opt.get("num_frame", 7)

        self.folders = []
        self.indices = []

        with open(opt["meta_info_file"], "r") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) < 4:
                    continue

                folder_name = parts[0]
                start, end, step = int(parts[1]), int(parts[2]), int(parts[3])
                frame_indices = list(range(start, end + 1, step))

                # 【核心修改】：自动过滤掉 meta_info 中不足 7 帧的行，防止 randint 报错
                if self.phase == "train" and len(frame_indices) < self.num_frame:
                    continue

                real_folder_name = folder_name.rsplit("_p", 1)[0]
                self.folders.append((folder_name, real_folder_name))
                self.indices.append(frame_indices)

        logger = get_root_logger()
        logger.info(f"DSEC {self.phase} Dataset Ready. Segments: {len(self.folders)}")

    def __getitem__(self, index):
        display_name, real_folder = self.folders[index]
        frame_list = self.indices[index]
        num_frames_target = (
            self.opt.get("num_frame", 7) if self.phase == "train" else len(frame_list)
        )

        if self.phase == "train":
            # 此时 frame_list 长度一定 >= 7，不会报错
            start_idx = random.randint(0, len(frame_list) - num_frames_target)
            selected_indices = frame_list[start_idx : start_idx + num_frames_target]

            gt_size = self.opt.get("gt_size", 256)
            lq_size = gt_size // self.scale
            h_lq, w_lq = 120, 160
            top = random.randint(0, h_lq - lq_size)
            left = random.randint(0, w_lq - lq_size)

            lq_shape, gt_shape = (3, lq_size, lq_size), (3, gt_size, gt_size)
            vx_shape = (self.num_bins, lq_size, lq_size)
        else:
            selected_indices = frame_list
            lq_shape, gt_shape = (3, 120, 160), (3, 480, 640)
            vx_shape = (self.num_bins, 120, 160)

        all_lq, all_gt, all_voxel = [], [], []

        for frame_idx in selected_indices:
            img_name = f"{frame_idx:06d}.png"
            gt_path = osp.join(self.gt_root, real_folder, img_name)
            lq_path = osp.join(self.lq_root, real_folder, img_name)
            voxel_path = osp.join(self.voxel_root, real_folder, f"{frame_idx:06d}.h5")

            img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)

            # 容错处理：如果硬盘读取瞬间失败，复制上一帧或补零
            if img_gt is None or img_lq is None or not osp.exists(voxel_path):
                if len(all_gt) > 0:
                    all_lq.append(all_lq[-1].clone())
                    all_gt.append(all_gt[-1].clone())
                    all_voxel.append(all_voxel[-1].clone())
                else:
                    all_lq.append(torch.zeros(lq_shape))
                    all_gt.append(torch.zeros(gt_shape))
                    all_voxel.append(torch.zeros(vx_shape))
                continue

            img_gt = cv2.resize(img_gt, (640, 480)).astype(np.float32) / 255.0
            img_lq = cv2.resize(img_lq, (160, 120)).astype(np.float32) / 255.0

            if self.phase == "train":
                img_lq = img_lq[top : top + lq_size, left : left + lq_size, :]
                img_gt = img_gt[
                    top * 4 : top * 4 + gt_size, left * 4 : left * 4 + gt_size, :
                ]
            else:
                img_gt = mod_crop(img_gt, self.scale)

            try:
                with h5py.File(voxel_path, "r", swmr=True) as f:
                    if self.phase == "train":
                        v_top, v_left = top * 4, left * 4
                        voxel_data = f["voxel_grid"][
                            :, v_top : v_top + gt_size, v_left : v_left + gt_size
                        ]
                    else:
                        voxel_data = f["voxel_grid"][()]

                    vx_raw = torch.from_numpy(voxel_data).float()
                    if vx_raw.shape[0] > self.num_bins:
                        vx_raw = vx_raw[: self.num_bins]
                    elif vx_raw.shape[0] < self.num_bins:
                        pad = torch.zeros((
                            self.num_bins - vx_raw.shape[0],
                            vx_raw.shape[1],
                            vx_raw.shape[2],
                        ))
                        vx_raw = torch.cat([vx_raw, pad], dim=0)

                    voxel_lq = F.interpolate(
                        vx_raw.unsqueeze(0),
                        size=(img_lq.shape[0], img_lq.shape[1]),
                        mode="area",
                    ).squeeze(0)
            except:
                voxel_lq = torch.zeros(vx_shape)

            all_lq.append(img2tensor(img_lq).clone().contiguous())
            all_gt.append(img2tensor(img_gt).clone().contiguous())
            all_voxel.append(voxel_lq.clone().contiguous())

        # 对齐长度
        while len(all_lq) < num_frames_target:
            all_lq.append(all_lq[-1].clone())
            all_gt.append(all_gt[-1].clone())
            all_voxel.append(all_voxel[-1].clone())

        return {
            "lq": torch.stack(all_lq, dim=0).contiguous(),
            "gt": torch.stack(all_gt, dim=0).contiguous(),
            "voxels": torch.stack(all_voxel, dim=0).contiguous(),
            "folder": display_name,
        }

    def __len__(self):
        return len(self.folders)
