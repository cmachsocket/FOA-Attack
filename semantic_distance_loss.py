"""
Semantic Distance Weighted Loss for Intermediate Layers

核心思路：
- 在初始化时（δ=0），计算每层 patch 和目标 patch 的平均余弦相似度
- dist_l = 1 - mean(cos(patch_adv_l[step=0], patch_tgt_l))
  → dist 越大 = 该层离目标越远 = 需要更大的更新力度
- alpha_l = base * sigmoid(w * dist_l + b)
  → 越不相似的层 alpha 越大，强制快速追赶
- 不依赖梯度一致性排序，在初始化时一次性计算，无额外训练开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class SemanticDistanceLoss(nn.Module):
    """
    语义距离加权损失

    每层的 alpha 由该层在初始化时与目标的语义距离决定。
    距离越远 → alpha 越大 → 强制追赶。

    与 Progressive Unfolding 的区别：
    - 不做层的时间展开，所有层全程参与
    - 通过 alpha 重加权来平衡各层的更新力度
    """

    def __init__(self,
                 extractors: List[nn.Module],
                 saliency_ratio: float = 0.3,
                 base_alpha: float = 0.5,
                 sigmoid_scale: float = 6.0,
                 local_weight: float = 0.2):
        super(SemanticDistanceLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)

        # 目标特征 per layer per model
        self.gt_global_per_layer = []   # list of {model_idx: [D]} per layer
        self.gt_local_per_layer = []    # list of {model_idx: [N, D]} per layer

        # 源图特征 per layer per model（用于计算 dist，只存一次）
        self.src_local_per_layer = []   # list of {model_idx: [N, D]} per layer

        self.layer_dist = None          # list of float

        self.saliency_ratio = saliency_ratio
        self.base_alpha = base_alpha
        self.sigmoid_scale = sigmoid_scale
        self.local_weight = local_weight
        self._dist_computed = False

    def _compute_layer_distances(self):
        """dist_l = 1 - mean(cos(patch_src[l], patch_tgt[l]))"""
        dist_per_layer = []
        num_layers = len(self.gt_local_per_layer)

        for li in range(num_layers):
            sim_list = []
            for model_idx in self.gt_local_per_layer[li]:
                src_local = self.src_local_per_layer[li][model_idx].unsqueeze(0)
                tgt_local = self.gt_local_per_layer[li][model_idx].unsqueeze(0)
                sim = F.cosine_similarity(src_local, tgt_local, dim=-1)
                sim_list.append(sim.mean().item())
            dist = 1.0 - np.mean(sim_list)
            dist_per_layer.append(dist)

        self.layer_dist = dist_per_layer
        self._dist_computed = True
        return dist_per_layer

    def _get_layer_alpha(self, layer_idx: int) -> float:
        if self.layer_dist is None:
            return self.base_alpha
        dist = self.layer_dist[layer_idx]
        alpha = self.base_alpha * (1.0 / (1.0 + np.exp(-self.sigmoid_scale * (dist - 0.5))))
        return alpha

    @torch.no_grad()
    def set_ground_truth(self, tgt_image: torch.Tensor, src_image: torch.Tensor = None):
        """
        每次调用都要重新提取目标特征。
        第一次调用时如果传了 src_image，顺便存下来并计算 dist。
        后续调用只需 tgt_image（src 已缓存）。
        """
        new_tgt_global = []
        new_tgt_local = []

        for model_idx, model in enumerate(self.extractors):
            _, all_global, all_local = model.intermediate_features(tgt_image.to(tgt_image.device))
            num_layers = len(all_global)

            # Extend lists if this model has more layers than previous models
            while len(new_tgt_global) < num_layers:
                new_tgt_global.append({})
                new_tgt_local.append({})

            for li in range(num_layers):
                new_tgt_global[li][model_idx] = all_global[li].squeeze(0)
                new_tgt_local[li][model_idx] = all_local[li].squeeze(0)

        # 第一次调用：存源图特征 + 计算 dist
        if src_image is not None and not self._dist_computed:
            src_global = []
            src_local = []
            for model_idx, model in enumerate(self.extractors):
                _, ag, al = model.intermediate_features(src_image.to(src_image.device))
                num_layers_src = len(ag)
                while len(src_global) < num_layers_src:
                    src_global.append({})
                    src_local.append({})
                for li in range(num_layers_src):
                    src_global[li][model_idx] = ag[li].squeeze(0)
                    src_local[li][model_idx] = al[li].squeeze(0)

            self.src_local_per_layer = src_local
            self.gt_global_per_layer = new_tgt_global
            self.gt_local_per_layer = new_tgt_local

            dist = self._compute_layer_distances()
            print(f"[SemanticDistance] Layer dist: {[f'{d:.3f}' for d in dist]}")
            print(f"[SemanticDistance] Layer alphas: {[f'{self._get_layer_alpha(li):.3f}' for li in range(len(dist))]}")
        else:
            # 后续调用：只更新目标特征
            self.gt_global_per_layer = new_tgt_global
            self.gt_local_per_layer = new_tgt_local

    def _compute_patch_loss(self, local_feat: torch.Tensor, tgt_local: torch.Tensor,
                            alpha: float) -> torch.Tensor:
        """显著性 Top-K 过滤 + 抑制重建"""
        feat_norm = torch.norm(local_feat, dim=-1)  # [B, N]
        num_patches = feat_norm.shape[-1]
        k = max(1, int(num_patches * self.saliency_ratio))
        _, topk_idx = torch.topk(feat_norm, k=k, dim=-1)  # [B, k]

        mask = torch.zeros(feat_norm.shape, dtype=torch.float32, device=feat_norm.device)
        mask.scatter_(1, topk_idx.long(), 1.0)

        reconstructed = local_feat * (1 - alpha * mask.unsqueeze(-1)) \
                      + tgt_local * (alpha * mask.unsqueeze(-1))

        sim = F.cosine_similarity(reconstructed, tgt_local, dim=-1)
        local_loss = ((1 - sim) * mask).sum() / (mask.sum() + 1e-8)
        return local_loss

    def forward(self,
                all_global_dict: Dict[int, List[torch.Tensor]],
                all_local_dict: Dict[int, List[torch.Tensor]],
                total_steps=None,
                y: any = None) -> torch.Tensor:
        """
        兼容 V1 接口：total_steps 可不传
        """
        loss_global_total = 0.0
        loss_local_total = 0.0
        num_active = 0

        for li in range(len(self.gt_global_per_layer)):
            layer_alpha = self._get_layer_alpha(li)
            for model_idx in range(len(self.extractors)):
                if model_idx not in self.gt_global_per_layer[li]:
                    continue
                if model_idx not in all_global_dict or li >= len(all_global_dict[model_idx]):
                    continue

                adv_g = all_global_dict[model_idx][li].unsqueeze(0)
                tgt_g = self.gt_global_per_layer[li][model_idx].unsqueeze(0)
                feat_loss = torch.sum(adv_g * tgt_g, dim=-1).mean()

                adv_local = all_local_dict[model_idx][li]
                tgt_local = self.gt_local_per_layer[li][model_idx]
                local_loss = self._compute_patch_loss(adv_local, tgt_local, layer_alpha)

                loss_global_total += feat_loss
                loss_local_total += local_loss
                num_active += 1

        num_active = max(num_active, 1)
        loss_global_total = loss_global_total / num_active
        loss_local_total = loss_local_total / num_active
        total_loss = loss_global_total + self.local_weight * loss_local_total
        return total_loss

    def get_layer_alphas(self) -> List[float]:
        if self.layer_dist is None:
            return [self.base_alpha] * max(len(self.gt_global_per_layer), 1)
        return [self._get_layer_alpha(li) for li in range(len(self.layer_dist))]