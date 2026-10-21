import torch.nn.functional as F


class SDESequence(Dataset):
    def __init__(self, sde_h5_path, num_bins=5, normalize=True):
        self.sde_h5_path = Path(sde_h5_path)
        self.num_bins = num_bins
        self.normalize = normalize
        # BFlow/RAFT 通常要求输入是 8 的倍数
        self.pad_to = 8

    def __getitem__(self, index):
        # ... (前面的读取逻辑保持不变)

        # 假设从 H5 读取到的 voxel 是 [5, 65, 86]
        h, w = voxel_curr.shape[-2:]

        # 计算需要 Padding 的数值
        pad_h = (self.pad_to - h % self.pad_to) % self.pad_to
        pad_w = (self.pad_to - w % self.pad_to) % self.pad_to

        # 进行右下角填充 (left, right, top, bottom)
        # 我们只填充右边和下面，这样左上角坐标是原生的
        padding = (0, pad_w, 0, pad_h)
        voxel_curr = F.pad(voxel_curr, padding, mode='constant', value=0)
        voxel_next = F.pad(voxel_next, padding, mode='constant', value=0)

        item = {
            'voxel_grid': voxel_curr,
            'voxel_grid_next': voxel_next,
            'meta': {
                'key': key_curr,
                'ori_shape': (h, w),  # 存下原始尺寸用于裁切
                'pad_shape': voxel_curr.shape[-2:]
            }
        }
        return item