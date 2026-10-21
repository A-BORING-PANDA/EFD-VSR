from functools import wraps
import torch
import torch.nn as nn
from torchvision import transforms
INTERP = transforms.InterpolationMode.BICUBIC



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def get_optical_flow_tile_mask(image_shape, tile_size):
    mask = torch.ones(image_shape, dtype=torch.bool)
    mask = make_tile_mask(mask, tile_size=tile_size)
    return mask

def make_tile_mask(mask, tile_size=1):
    n = tile_size
    s = n // 2
    tile_mask = torch.zeros_like(mask, dtype=torch.bool)
    tile_mask[s::n, s::n] = True
    return (mask & tile_mask)

def coeffs_grid_to_list(coeff_grid, mask, num_coeffs):
    """Convert a 2D grid of DCT coefficients to a list using a binary mask.

    Args:
        coeff_grid: Shape [b, s, k, h, w], where k represents coefficients for both x & y,
            and (h,w) is the image shape. s is the number scales.
        mask: Shape [h, w], a binary mask marking 'valid' trajectories.
        num_coeffs: The number of coefficients.
        patch_size: The pixel size of each tile having one trajectory.

    Returns:
        coeff_list [b, 2, n, num_coeffs], pixel_positiosn, orig_shape
    """
    coeffs, pixel_positions, orig_shape = grid_to_list(coeff_grid, mask)
    b, s, c2, h, w = orig_shape
    assert c2 == 2 * num_coeffs
    coeffs = coeffs.reshape(b, s, 2, num_coeffs, -1)
    return coeffs.permute(0, 1, 2, 4, 3), pixel_positions, orig_shape

def grid_to_list(image_grid: torch.Tensor, object_mask: torch.Tensor):
    """Return all entries/features of an image grid as sequential list,
    where it only keeps one for each patch of shape patch_size x patch_size,
    and additionally only considers entries in the object mask.

    Args:
        image_grid: [b, s, c, h, w]
        object_mask: [h, w]

    Returns:
        [b, s, c, n], where n is the number of true pixels in mask.
    """
    pixel_positions = torch.nonzero(object_mask)
    mask = object_mask.reshape(-1)
    orig_shape = image_grid.shape
    b, s, c, h, w = orig_shape
    pixel_list = image_grid.reshape(b, s, c, -1)
    pixel_list = pixel_list[..., mask]
    return pixel_list, pixel_positions, orig_shape

def dense_flow_from_traj(traj_flow, pixel_positions, patch_size, image_shape):
    h, w = image_shape
    grid_h, grid_w = h // patch_size, w // patch_size
    patch_flow = list_to_grid(traj_flow, pixel_positions // patch_size,
                             (grid_h, grid_w))
    return interpolate_dense_flow(patch_flow, image_shape), patch_flow

def interpolate_dense_flow(patch_flow, orig_image_shape):
    return transforms.functional.resize(patch_flow, orig_image_shape,
                                              interpolation=INTERP, antialias=True)

def list_to_grid(feature_list, pixel_positions, image_shape):
    """Put features of dimension c into the correct spatial positions

    Args:
        feature_list: [b, n, c]
        pixel_positions: [n, 2]
        image_shape: (height, width)

    Returns:
        feature_grid: [b, c, h, w]
    """
    b, n, c = feature_list.shape
    h, w = image_shape
    coeff_grid = torch.zeros((b, c, h, w), dtype=feature_list.dtype,
                             device=feature_list.device)

    batch_indices = torch.arange(b, dtype=torch.long, device=feature_list.device)[:, None]
    y_indices, x_indices = pixel_positions[:, 0], pixel_positions[:, 1]

    coeff_grid[batch_indices, :, y_indices, x_indices] = feature_list
    return coeff_grid
