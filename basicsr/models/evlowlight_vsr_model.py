import torch
import os
from collections import Counter, OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.data.transforms import mod_crop
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class EvLowLightVSR_Model(VideoBaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.scale = opt["scale"]
        if self.is_train:
            self.fix_flow_iter = opt["train"].get("fix_flow")

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        # 适配 DSEC 的 voxels
        if "voxels" in data:
            self.voxels = data["voxels"].to(self.device)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """重写 validation 方法：专门用于提取并保存光流和中间特征"""
        dataset = dataloader.dataset
        dataset_name = dataset.opt["name"]
        rank, world_size = get_dist_info()

        if rank == 0:
            pbar = tqdm(total=len(dataset), unit="folder")
            logger = get_root_logger()
            logger.info(f"Start extracting features for dataset: {dataset_name}...")

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size

        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data["folder"]

            # 补充 batch 维度
            val_data["lq"].unsqueeze_(0)
            if "gt" in val_data:
                val_data["gt"].unsqueeze_(0)
            if "voxels" in val_data:
                val_data["voxels"].unsqueeze_(0)

            self.feed_data(val_data)

            # 还原字典形状
            val_data["lq"].squeeze_(0)
            if "gt" in val_data:
                val_data["gt"].squeeze_(0)
            if "voxels" in val_data:
                val_data["voxels"].squeeze_(0)

            # 获取网络模型 (支持 EMA)
            net = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
            net.eval()
            bare_net = net.module if hasattr(net, "module") else net

            # 提取特征与光流
            with torch.no_grad():
                flows, events, encoder1_output, decoder2_output, dcn_info = bare_net.extract_flow_and_features(self.lq, self.voxels)

            # 保存特征字典至 visualization/DSEC_Test/features/视频文件夹/
            if rank == 0 and i < num_folders:
                feat_save_dir = osp.join(self.opt["path"]["visualization"], dataset_name, "features", folder)
                os.makedirs(feat_save_dir, exist_ok=True)

                # ==================== 🛠️ 修复核心：安全转换 offsets 和 masks 到 CPU ====================
                offsets_cpu = [v.cpu() if v is not None else None for v in dcn_info['offsets']] if dcn_info and dcn_info.get('offsets') else None
                masks_cpu = [v.cpu() if v is not None else None for v in dcn_info['masks']] if dcn_info and dcn_info.get('masks') else None

                save_dict = {
                    'flows': {k: [v.cpu() for v in val_list] for k, val_list in flows.items() if isinstance(val_list, list)},
                    'events': {k: [v.cpu() for v in val_list] for k, val_list in events.items() if isinstance(val_list, list)},
                    'encoder1_output': [v.cpu() for v in encoder1_output],
                    'decoder2_output': [v.cpu() for v in decoder2_output],
                    'dcn_offsets': offsets_cpu,
                    'dcn_masks': masks_cpu
                }
                save_path = osp.join(feat_save_dir, "sequence_features.pt")
                torch.save(save_dict, save_path)

            # 显存清理
            del self.lq
            if hasattr(self, "voxels"):
                del self.voxels
            if hasattr(self, "gt"):
                del self.gt
            torch.cuda.empty_cache()

            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f"Extracted: {folder}")

        if rank == 0:
            pbar.close()

        if self.opt["dist"]:
            dist.barrier()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info("Running nondist_validation for feature extraction...")
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)