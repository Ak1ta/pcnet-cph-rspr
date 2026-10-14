from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .alignment.network import IHN
from .alignment.utils import warp
from .backbones.swin import SwinTransformer
from .config import IHNConfig, ModelConfig


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def generate_valid_warp_mask(warped_thermal: torch.Tensor) -> torch.Tensor:
    valid = (warped_thermal != 0).float().sum(dim=1, keepdim=True)
    return (valid > 0).float()


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError("Unsupported activation: {}".format(name))


class GMSALayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        rgb_features: Tensor,
        thermal_features: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        attended = self.multihead_attn(
            query=self.with_pos_embed(rgb_features, query_pos).transpose(0, 1),
            key=self.with_pos_embed(thermal_features, pos).transpose(0, 1),
            value=thermal_features.transpose(0, 1),
            need_weights=False,
        )[0].transpose(0, 1)
        rgb_features = self.norm2(rgb_features + self.dropout2(attended))
        feedforward = self.linear2(
            self.dropout(self.activation(self.linear1(rgb_features)))
        )
        return self.norm3(rgb_features + self.dropout3(feedforward))


class GMSA(nn.Module):
    def __init__(self, d_model=256, num_layers=2, decoder_layer=None):
        super().__init__()
        if decoder_layer is None:
            decoder_layer = GMSALayer(d_model=d_model, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, rgb_features: Tensor, thermal_features: Tensor) -> Tensor:
        output = rgb_features
        for layer in self.layers:
            output = layer(output, thermal_features)
        return output


class ChallengeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_attributes: int, dropout: float):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_attributes)

    def forward(self, rgb_deep: Tensor, thermal_deep: Tensor):
        rgb_global = F.adaptive_avg_pool2d(rgb_deep, 1).flatten(1)
        thermal_global = F.adaptive_avg_pool2d(thermal_deep, 1).flatten(1)
        context = self.context(torch.cat((rgb_global, thermal_global), dim=1))
        return self.classifier(context)


class ConvBNGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class RegionStructurePrototypeRectifier(nn.Module):
    """RSPR residual refinement with saliency-GT-derived boundary supervision."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.rspr_channels
        self.grid_size = config.rspr_grid_size
        self.gamma = config.rspr_confidence_gamma
        self.epsilon = config.rspr_epsilon
        self.rgb_detail = ConvBNGELU(128, channels, 1)
        self.structure_embedding = ConvBNGELU(64 + channels, channels, 3)
        self.edge_head = nn.Sequential(
            ConvBNGELU(channels, max(channels // 2, 8), 3),
            nn.Conv2d(max(channels // 2, 8), 1, kernel_size=1),
        )
        self.correction_features = ConvBNGELU(channels + 3, channels, 3)
        self.correction_out = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.correction_out.weight)
        nn.init.zeros_(self.correction_out.bias)

    def _weighted_prototypes(self, features: Tensor, weights: Tensor):
        grid = (self.grid_size, self.grid_size)
        local_numerator = F.adaptive_avg_pool2d(features * weights, grid)
        local_denominator = F.adaptive_avg_pool2d(weights, grid)
        local = local_numerator / local_denominator.clamp_min(self.epsilon)
        global_numerator = (features * weights).sum(dim=(2, 3), keepdim=True)
        global_denominator = weights.sum(dim=(2, 3), keepdim=True)
        global_prototype = global_numerator / global_denominator.clamp_min(
            self.epsilon
        )
        prototypes = torch.cat(
            (global_prototype.flatten(2), local.flatten(2)), dim=2
        )
        local_valid = local_denominator.flatten(2) > self.epsilon
        global_valid = torch.ones(
            weights.shape[0], 1, 1, dtype=torch.bool, device=weights.device
        )
        valid = torch.cat((global_valid, local_valid), dim=2)
        return F.normalize(prototypes, dim=1, eps=self.epsilon), valid

    def _prototype_affinity(
        self, features: Tensor, coarse_probability: Tensor
    ) -> Tensor:
        probability = F.interpolate(
            coarse_probability.detach(),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        foreground, foreground_valid = self._weighted_prototypes(
            features, probability.pow(self.gamma)
        )
        background, background_valid = self._weighted_prototypes(
            features, (1.0 - probability).pow(self.gamma)
        )
        normalized = F.normalize(features, dim=1, eps=self.epsilon)
        foreground_similarity = torch.einsum(
            "bchw,bck->bkhw", normalized, foreground
        ).masked_fill(
            ~foreground_valid.squeeze(1).unsqueeze(-1).unsqueeze(-1), -1e4
        )
        background_similarity = torch.einsum(
            "bchw,bck->bkhw", normalized, background
        ).masked_fill(
            ~background_valid.squeeze(1).unsqueeze(-1).unsqueeze(-1), -1e4
        )
        return (
            foreground_similarity.max(dim=1, keepdim=True).values
            - background_similarity.max(dim=1, keepdim=True).values
        )

    def forward(
        self, decoder_feature: Tensor, rgb_shallow: Tensor, coarse_logits: Tensor
    ) -> Dict[str, Tensor]:
        detail = self.rgb_detail(rgb_shallow)
        structure = self.structure_embedding(
            torch.cat((decoder_feature, detail), dim=1)
        )
        affinity = self._prototype_affinity(structure, torch.sigmoid(coarse_logits))
        edge_low = self.edge_head(structure)
        coarse_low = F.interpolate(
            coarse_logits,
            size=structure.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        correction_features = self.correction_features(
            torch.cat(
                (structure, affinity, torch.sigmoid(edge_low), coarse_low), dim=1
            )
        )
        correction = F.interpolate(
            self.correction_out(correction_features),
            size=coarse_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        edge = F.interpolate(
            edge_low,
            size=coarse_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "saliency": coarse_logits + correction,
            "edge": edge,
            "prototype_affinity": affinity,
            "correction": correction,
        }


class PCNet(nn.Module):
    """One PCNet implementation composed with only the requested innovations."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig(experiment="pcnet")
        self.config = config
        self.rgb_swin = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            use_checkpoint=self.config.activation_checkpoint,
        )
        self.t_swin = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            use_checkpoint=self.config.activation_checkpoint,
        )

        self.IHN = IHN(args=IHNConfig(self.config))
        self._set_ihn_adapter_policy()
        self.IHN.eval()

        self.MSA_sem = GMSA(d_model=1024)
        self.conv_sem = conv3x3_bn_relu(2048, 1024)
        self.conv_sem3 = conv3x3_bn_relu(1024, 1024)
        self.conv_sem2 = conv3x3_bn_relu(1024, 512)
        self.conv_sem1 = conv3x3_bn_relu(1024, 256)

        self.MSA4_r = GMSA(d_model=1024)
        self.MSA3_r = GMSA(d_model=512)
        self.MSA2_r = GMSA(d_model=256)
        self.MSA4_2_r = GMSA(d_model=1024)
        self.MSA3_2_r = GMSA(d_model=512)
        self.MSA2_2_r = GMSA(d_model=256)

        self.conv_attr4 = conv3x3_bn_relu(1024, 1024)
        self.conv_attr3 = conv3x3_bn_relu(512, 512)
        self.conv_attr2 = conv3x3_bn_relu(256, 256)
        self.conv_fr4 = conv3x3_bn_relu(1024, 1024)
        self.conv_fr3 = conv3x3_bn_relu(512, 512)
        self.conv_fr2 = conv3x3_bn_relu(256, 256)

        self.convAtt4 = conv3x3_bn_relu(1024, 1024)
        self.convAtt3 = conv3x3_bn_relu(512, 512)
        self.convAtt2 = conv3x3_bn_relu(256, 256)
        self.convAtt1 = conv3x3_bn_relu(128, 128)
        self.conv1024 = conv3x3_bn_relu(1024, 512)
        self.conv512 = conv3x3_bn_relu(512, 256)
        self.conv256 = conv3x3_bn_relu(256, 128)
        self.conv128 = conv3x3_bn_relu(128, 64)
        self.conv64 = conv3x3(64, 1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        # Innovations are instantiated only when the experiment uses them.
        # Each branch has its own deterministic RNG stream, so adding another
        # branch cannot perturb the PCNet or sibling-branch initialization.
        self.challenge_head = None
        self.rspr = None
        if self.config.spec.challenge_perception:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.config.seed + 10_001)
                self.challenge_head = ChallengeHead(
                    self.config.attribute_hidden_dim,
                    self.config.num_attributes,
                    self.config.dropout,
                )
        if self.config.spec.rspr:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.config.seed + 30_001)
                self.rspr = RegionStructurePrototypeRectifier(self.config)

    def _set_ihn_adapter_policy(self) -> None:
        for parameter in self.IHN.parameters():
            parameter.requires_grad = False
        for name, parameter in self.IHN.fnet1.named_parameters():
            parameter.requires_grad = "Adapter" in name

    @staticmethod
    def _tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)

    def _run_challenge_branch(self, features_rgb, features_thermal):
        if self.challenge_head is None:
            return None, None
        logits = self.challenge_head(features_rgb[3], features_thermal[3])
        return logits, torch.sigmoid(logits)

    def forward(
        self,
        rgb: Tensor,
        thermal: Tensor,
        ihn_rgb: Tensor,
        ihn_thermal: Tensor,
    ) -> Dict[str, object]:
        fr = self.rgb_swin(rgb)
        ft = self.t_swin(thermal)
        attribute_logits, attribute_probabilities = self._run_challenge_branch(fr, ft)

        height4, width4 = fr[3].shape[-2:]
        semantic_tokens = torch.cat(
            (fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2)),
            dim=1,
        )
        semantic_warp = self.MSA_sem(semantic_tokens, semantic_tokens)
        semantic_rgb, semantic_t = torch.split(
            semantic_warp, height4 * width4, dim=1
        )
        semantic_warp = self.conv_sem(
            torch.cat(
                (
                    self._tokens_to_map(semantic_rgb, height4, width4),
                    self._tokens_to_map(semantic_t, height4, width4),
                ),
                dim=1,
            )
        )
        semantic_warp3 = self.conv_sem3(self.up1(semantic_warp))
        semantic_warp2 = self.conv_sem2(self.up2(semantic_warp))
        semantic_warp1 = self.conv_sem1(self.up4(semantic_warp))

        four_point_disp, coords1, coords0 = self.IHN(
            ihn_rgb, ihn_thermal, semantic_warp
        )
        flow = coords1 - coords0
        flow = F.interpolate(flow, scale_factor=12, mode="bilinear", align_corners=True) * 12
        warped_thermal = warp(thermal, flow)
        valid_mask = generate_valid_warp_mask(warped_thermal)

        fr1_masked = fr[1] * F.interpolate(
            valid_mask, size=fr[1].shape[-2:], mode="bilinear", align_corners=False
        )
        fr2_masked = fr[2] * F.interpolate(
            valid_mask, size=fr[2].shape[-2:], mode="bilinear", align_corners=False
        )
        fr3_masked = fr[3] * F.interpolate(
            valid_mask, size=fr[3].shape[-2:], mode="bilinear", align_corners=False
        )

        att4 = self.MSA4_r(
            (fr3_masked * semantic_warp3).flatten(2).transpose(1, 2),
            ft[3].flatten(2).transpose(1, 2),
        )
        att3 = self.MSA3_r(
            (fr2_masked * semantic_warp2).flatten(2).transpose(1, 2),
            ft[2].flatten(2).transpose(1, 2),
        )
        att2 = self.MSA2_r(
            (fr1_masked * semantic_warp1).flatten(2).transpose(1, 2),
            ft[1].flatten(2).transpose(1, 2),
        )

        cross4 = self.conv_attr4(self._tokens_to_map(att4, *fr[3].shape[-2:]))
        cross3 = self.conv_attr3(self._tokens_to_map(att3, *fr[2].shape[-2:]))
        cross2 = self.conv_attr2(self._tokens_to_map(att2, *fr[1].shape[-2:]))
        fused4 = self.conv_fr4(fr[3] + cross4)
        fused3 = self.conv_fr3(fr[2] + cross3)
        fused2 = self.conv_fr2(fr[1] + cross2)

        intra4 = self.MSA4_2_r(
            fused4.flatten(2).transpose(1, 2), fused4.flatten(2).transpose(1, 2)
        )
        intra3 = self.MSA3_2_r(
            fused3.flatten(2).transpose(1, 2), fused3.flatten(2).transpose(1, 2)
        )
        intra2 = self.MSA2_2_r(
            fused2.flatten(2).transpose(1, 2), fused2.flatten(2).transpose(1, 2)
        )

        r4 = self.convAtt4(self._tokens_to_map(intra4, *fused4.shape[-2:]))
        r3 = self.convAtt3(self._tokens_to_map(intra3, *fused3.shape[-2:]))
        r2 = self.convAtt2(self._tokens_to_map(intra2, *fused2.shape[-2:]))
        r1 = self.convAtt1(fr[0])

        r4 = self.conv1024(self.up2(r4))
        r3 = self.conv512(self.up2(r3 + r4))
        r2 = self.conv256(self.up2(r2 + r3))
        r1 = self.conv128(r1 + r2)
        coarse_output = self.conv64(self.up4(r1))
        output = coarse_output
        edge = None
        prototype_affinity = None
        correction = None
        if self.rspr is not None:
            rectified = self.rspr(r1, fr[0], coarse_output)
            output = rectified["saliency"]
            edge = rectified["edge"]
            prototype_affinity = rectified["prototype_affinity"]
            correction = rectified["correction"]
        return {
            "saliency": output,
            "coarse_saliency": coarse_output,
            "edge": edge,
            "prototype_affinity": prototype_affinity,
            "rspr_correction": correction,
            "attributes": attribute_logits,
            "attribute_probabilities": attribute_probabilities,
            "challenge_active": self.challenge_head is not None,
            "rspr_active": self.rspr is not None,
            "homography_displacement": four_point_disp,
        }

    def load_pretrained_components(self, swin_path: str, ihn_path: str) -> Dict[str, object]:
        swin_checkpoint = torch.load(swin_path, map_location="cpu")
        swin_state_dict = swin_checkpoint.get("model", swin_checkpoint)
        rgb_result = self.rgb_swin.load_state_dict(swin_state_dict, strict=False)
        thermal_result = self.t_swin.load_state_dict(swin_state_dict, strict=False)
        ihn_state_dict = torch.load(ihn_path, map_location="cpu")
        ihn_result = self.IHN.load_state_dict(ihn_state_dict, strict=False)

        def summarize(module: nn.Module, result) -> Dict[str, object]:
            missing = tuple(result.missing_keys)
            unexpected = tuple(result.unexpected_keys)
            matched = len(module.state_dict()) - len(missing)
            if matched <= 0:
                raise RuntimeError(
                    "A pretrained checkpoint matched no tensors in {}.".format(
                        type(module).__name__
                    )
                )
            return {
                "matched_tensors": matched,
                "missing_count": len(missing),
                "unexpected_count": len(unexpected),
                "missing_examples": missing[:20],
                "unexpected_examples": unexpected[:20],
            }

        return {
            "rgb_swin": summarize(self.rgb_swin, rgb_result),
            "thermal_swin": summarize(self.t_swin, thermal_result),
            "ihn": summarize(self.IHN, ihn_result),
        }

    def count_branch_parameters(self) -> Dict[str, int]:
        def count(module: Optional[nn.Module]) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        return {
            "cph": count(self.challenge_head),
            "rspr": count(self.rspr),
        }
