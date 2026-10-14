from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Exact smoothing convention used by the released PCNet code."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = targets.size(0)
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        score = 2.0 * (intersection + self.smooth) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - score.sum() / batch_size


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Attribute logits/targets differ: {} vs {}.".format(
                    tuple(logits.shape), tuple(targets.shape)
                )
            )
        targets = targets.to(dtype=logits.dtype)
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        loss = targets * torch.log(positive.clamp_min(self.eps))
        loss += (1.0 - targets) * torch.log(negative.clamp_min(self.eps))
        probability = positive * targets + negative * (1.0 - targets)
        gamma = self.gamma_positive * targets + self.gamma_negative * (1.0 - targets)
        loss *= torch.pow(1.0 - probability, gamma)
        return -loss.mean()


def saliency_smoothness(
    prediction: torch.Tensor, target: torch.Tensor, alpha: float = 10.0
) -> torch.Tensor:
    sobel_x = prediction.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).view(1, 1, 3, 3)
    sobel_y = prediction.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ).view(1, 1, 3, 3)
    saliency_x = F.conv2d(prediction, sobel_x, padding=1).abs()
    saliency_y = F.conv2d(prediction, sobel_y, padding=1).abs()
    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)
    weight_x = torch.exp(-alpha * target_x.abs())
    weight_y = torch.exp(-alpha * target_y.abs())
    epsilon = 0.001
    penalty_x = torch.sqrt((saliency_x * weight_x).pow(2) + epsilon**2)
    penalty_y = torch.sqrt((saliency_y * weight_y).pow(2) + epsilon**2)
    return 10.0 * (penalty_x + penalty_y).mean()


def saliency_loss(
    logits: torch.Tensor, targets: torch.Tensor, dice: SoftDiceLoss
) -> Dict[str, torch.Tensor]:
    logits = logits.float()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice_value = dice(logits, targets)
    smooth = saliency_smoothness(torch.sigmoid(logits), targets)
    return {
        "total": bce + dice_value + smooth,
        "bce": bce,
        "dice": dice_value,
        "smoothness": smooth,
    }


def dilated_boundary(target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    target = target.clamp(0.0, 1.0)
    if kernel_size == 1:
        return target
    return F.max_pool2d(
        target, kernel_size, stride=1, padding=kernel_size // 2
    )


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice: SoftDiceLoss,
    positive_weight_cap: float,
) -> Dict[str, torch.Tensor]:
    positive = target.sum().detach()
    negative = target.new_tensor(target.numel()) - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(
        1.0, positive_weight_cap
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight.reshape(1, 1, 1)
    )
    dice_value = dice(logits, target)
    return {
        "total": bce + dice_value,
        "bce": bce,
        "dice": dice_value,
        "positive_weight": positive_weight,
    }


def clip_gradients(optimizer: torch.optim.Optimizer, margin: float) -> None:
    """Released PCNet element-wise clipping, not global-norm clipping."""

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.data.clamp_(-margin, margin)
