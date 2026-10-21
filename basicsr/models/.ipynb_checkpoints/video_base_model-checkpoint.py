import torch
import os
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.utils.dist_util import get_dist_info
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel

@MODEL_REGISTRY.register()
class VideoBaseModel(SRModel):
    """Base video SR model (Modified strictly for Feature & Flow Extraction)."""

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Modified validation to only extract and save network intermediate features."""
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        rank, world_size = get_dist_info()

        # ==================== 1. 初始化进度条 ====================
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='frame')
            logger = get_root_logger()
            logger.info(f"Start extracting features for dataset: {dataset_name}...")

        # ==================== 2. 遍历数据集 ====================
        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]
            # 补齐 batch 维度
            val_data['lq'].unsqueeze_(0)
            if 'gt' in val_data:
                val_data['gt'].unsqueeze_(0)
            
            folder = val_data['folder']
            frame_idx, max_idx = val_data['idx'].split('/')
            
            # 将数据送入 GPU
            self.feed_data(val_data)
            
            # 获取网络模型 (支持 EMA)
            if hasattr(self, 'net_g_ema'):
                net = self.net_g_ema
            else:
                net = self.net_g

            net.eval()
            
            # ==================== 3. 提取特征 ====================
            with torch.no_grad():
                if hasattr(self, 'voxel'):
                    voxel_data = self.voxel
                elif hasattr(self, 'lq_voxel'):
                    voxel_data = self.lq_voxel
                else:
                    voxel_data = val_data.get('voxel', val_data.get('lq_voxel', None))
                    if voxel_data is not None:
                        voxel_data = voxel_data.to(self.device).unsqueeze(0)

                if hasattr(net, 'extract_flow_and_features'):
                    flows, events, encoder1_output, decoder2_output, dcn_info = net.extract_flow_and_features(self.lq, voxel_data)
                else:
                    raise NotImplementedError("网络模型中未找到 extract_flow_and_features 方法！")

            # ==================== 4. 保存特征 ====================
            if rank == 0:
                feat_save_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'features', folder)
                os.makedirs(feat_save_dir, exist_ok=True)
                
                # 安全将 DCN offset 和 mask 转到 CPU
                offsets_cpu = [v.cpu() if v is not None else None for v in dcn_info['offsets']] if dcn_info['offsets'] else None
                masks_cpu = [v.cpu() if v is not None else None for v in dcn_info['masks']] if dcn_info['masks'] else None

                save_dict = {
                    'flows': {k: [v.cpu() for v in val_list] for k, val_list in flows.items() if isinstance(val_list, list)},
                    'events': {k: [v.cpu() for v in val_list] for k, val_list in events.items() if isinstance(val_list, list)},
                    'encoder1_output': [v.cpu() for v in encoder1_output],
                    'decoder2_output': [v.cpu() for v in decoder2_output],
                    'dcn_offsets': offsets_cpu,
                    'dcn_masks': masks_cpu
                }
                save_path = osp.join(feat_save_dir, f'frame_{int(frame_idx):04d}_feats.pt')
                torch.save(save_dict, save_path)

            # ==================== 5. 显存清理 ====================
            # 及时清理大张量，防止长时间运行爆显存
            del self.lq
            if hasattr(self, 'gt'):
                del self.gt
            if 'voxel_data' in locals():
                del voxel_data
            torch.cuda.empty_cache()

            # 更新进度条
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Extracting {folder}: {int(frame_idx) + world_size}/{max_idx}')

        # ==================== 6. 结束 ====================
        if rank == 0:
            pbar.close()
            
        if self.opt['dist']:
            dist.barrier()
        
        # 删除原本计算 metrics 和记录 logger 的逻辑，因为我们不再评估图像质量。

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Running nondist_validation for feature extraction...')
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)