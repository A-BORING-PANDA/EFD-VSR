import torch
import torch.nn as nn
import numpy as np

from basicsr.archs.unet_arch import UNet
from basicsr.utils.registry import ARCH_REGISTRY
from .flownet_util import (
    initialize_weights,
    get_optical_flow_tile_mask,
    coeffs_grid_to_list,
    dense_flow_from_traj,
)


@ARCH_REGISTRY.register()
class TrajectoryNet(nn.Module):
    def __init__(
        self,
        image_shape=(120, 160),
        num_bins=15,
        num_basis=5,
        patch_size=4,
        basis_type="polynomial",
        skip_frames=1,
    ):
        super(TrajectoryNet, self).__init__()
        self.image_shape = image_shape
        self.num_basis = num_basis
        self.patch_size = patch_size
        self.basis_type = basis_type
        self.skip_frames = skip_frames
        self.anchor_time = 0

        self.model = UNet(num_bins, 2 * num_basis)
        initialize_weights(self.model)

        if self.basis_type in ["learned", "learned_2d"]:
            n_out = num_basis if basis_type == "learned" else 2 * num_basis
            self.basis_network = nn.Sequential(
                nn.Linear(1, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, n_out),
            )
            initialize_weights(self.basis_network)
        else:
            self.basis_network = None

        # 3. 注册 Mask 缓存
        object_mask = get_optical_flow_tile_mask(image_shape, patch_size)
        self.register_buffer("object_mask", object_mask, persistent=False)

    def compute_basis(self, coeffs, times):
        """Calculate trajectory values for given coefficients at n_t
        given times.

        Args:
            coeffs: [b, s, 2, n, k]
            times: [1, n_t]

        Raises:
            ValueError: _description_

        Returns:
            [b, n, k, n_t, 2]
        """
        if self.basis_type == "dct":
            K = self.num_basis
            T = 1
            k_idx = torch.arange(1, K + 1, device=coeffs.device)
            A = (2 * times[..., None] + 1) * k_idx[None, None, :]
            in_cos = np.pi / (2.0 * T) * A
            basis = np.sqrt(2.0 / T) * torch.cos(in_cos)

        elif self.basis_type == "learned":
            basis = self.basis_network(times[..., None])  # [b, m, k]

        elif self.basis_type == "polynomial":
            k_idx = torch.arange(1, self.num_basis + 1, device=coeffs.device)
            basis = times[..., None] ** k_idx[None, None, :]
        else:
            raise ValueError

        raw_coeff_y = coeffs[..., 1, :, :]  # [b, n, k]
        raw_coeff_x = coeffs[..., 0, :, :]  # [b, n, k]
        # 这里和spynet的索引不一样，已修改
        y_product = basis[..., None, :, :] * raw_coeff_y[..., None, :]  # [n, m, k]
        x_product = basis[..., None, :, :] * raw_coeff_x[..., None, :]  # [n, m, k]

        coords = torch.stack(
            [
                torch.sum(x_product, dim=-1),
                torch.sum(y_product, dim=-1),
            ],
            dim=-1,
        )

        return torch.sum(coords, dim=1)  # Sum trajectories of all scales

    def calculate_flow(self, coeff_grid, mask):
        # 核心接口：输入网络预测的网格，输出 (B, 2, H, W) 的光流图
        coeffs, pixel_positions, _ = coeffs_grid_to_list(
            coeff_grid, mask, num_coeffs=self.num_basis
        )

        ts = torch.tensor(
            self.anchor_time, device=coeff_grid.device, dtype=coeff_grid.dtype
        )[None]
        te = torch.tensor(
            1 / self.skip_frames, device=coeff_grid.device, dtype=coeff_grid.dtype
        )[None]

        traj_at_ts = self.compute_basis(coeffs, ts)
        traj_at_te = self.compute_basis(coeffs, te)
        traj_flow = (traj_at_te - traj_at_ts)[:, :, 0, :]
        h, w = mask.shape
        # 将轨迹位移还原为稠密光流
        flows, _ = dense_flow_from_traj(
            traj_flow,  # 传入 [b, n, 2]
            pixel_positions,
            self.patch_size,
            (h, w),
        )
        return flows

    def forward(self, x):
        b, c, h, w = x.shape
        curr_mask = get_optical_flow_tile_mask((h, w), self.patch_size).to(x.device)
        # 前向传播得到系数网格，并直接转换为光流
        coeff_grid = self.model(x)
        if len(coeff_grid.shape) == 4:
            coeff_grid = coeff_grid[:, None]
        return self.calculate_flow(coeff_grid, curr_mask)
