import torch
import torch.nn.functional as F


def bilinear_sampler(image, coordinates, mode="bilinear", mask=False):
    """Sample an image using pixel-space coordinates."""

    height, width = image.shape[-2:]
    x_grid, y_grid = coordinates.split([1, 1], dim=-1)
    x_grid = 2 * x_grid / (width - 1) - 1
    y_grid = 2 * y_grid / (height - 1) - 1
    grid = torch.cat([x_grid, y_grid], dim=-1)
    sampled = F.grid_sample(image, grid, mode=mode, align_corners=True)
    if mask:
        valid = (
            (x_grid > -1) & (y_grid > -1) & (x_grid < 1) & (y_grid < 1)
        )
        return sampled, valid.float()
    return sampled


def coords_grid(batch, height, width):
    try:
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
    except TypeError:  # PyTorch 1.x compatibility
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    coordinates = torch.stack((x_grid, y_grid), dim=0).float()
    return coordinates[None].expand(batch, -1, -1, -1)


def warp(tensor, flow):
    """Warp a tensor according to optical flow using the released convention."""

    batch, _, height, width = tensor.size()
    x_grid = torch.arange(0, width, device=tensor.device).view(1, -1)
    x_grid = x_grid.repeat(height, 1)
    y_grid = torch.arange(0, height, device=tensor.device).view(-1, 1)
    y_grid = y_grid.repeat(1, width)
    x_grid = x_grid.view(1, 1, height, width).repeat(batch, 1, 1, 1)
    y_grid = y_grid.view(1, 1, height, width).repeat(batch, 1, 1, 1)
    grid = torch.cat((x_grid, y_grid), 1).to(dtype=tensor.dtype)
    sample_grid = grid + flow
    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / max(width - 1, 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / max(height - 1, 1) - 1.0
    sample_grid = sample_grid.permute(0, 2, 3, 1)
    output = F.grid_sample(tensor, sample_grid, align_corners=True)
    valid = F.grid_sample(
        torch.ones_like(tensor), sample_grid, align_corners=True
    )
    valid = (valid >= 0.999).to(dtype=tensor.dtype)
    return output * valid
