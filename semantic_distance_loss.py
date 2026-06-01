"""
Semantic Distance Weighted Loss for Intermediate Layers

核心思路：
- 在初始化时（δ=0），计算每层 patch 和目标 patch 的平均余弦相似度
- dist_l = 1 - mean(cos(patch_adv_l[step=0], patch_tgt_l))
  → dist 越大 = 该层离目标越远 = 需要更大的更新力度
- alpha_l = base * sigmoid(w * dist_l + b)
  → 越不相似的层 alpha 越大，强制快速追赶
- 每步 forward 都重新计算当前层的 dist，用 EMA 平滑噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class SemanticDistanceLoss(nn.Module):
    """
    语义距离加权损失

    每层的 alpha 由该层当前与目标的语义距离动态决定（每步更新）。
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
                 local_weight: float = 0.2,
                 ema_beta: float = 0.9):
        super(SemanticDistanceLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)

        # 目标特征 per layer per model
        self.gt_global_per_layer = []   # list of {model_idx: [D]} per layer
        self.gt_local_per_layer = []    # list of {model_idx: [N, D]} per layer

        # 源图特征 per layer per model（用于计算 dist，只存一次）
        self.src_local_per_layer = []   # list of {model_idx: [N, D]} per layer

        self.saliency_ratio = saliency_ratio
        self.base_alpha = base_alpha
        self.sigmoid_scale = sigmoid_scale
        self.local_weight = local_weight
        self.ema_beta = ema_beta

        # EMA 平滑后的每层 dist（每步更新）
        self.dist_ema = None   # list of float

        # 初始化时的静态 dist（仅用于日志/调试）
        self.layer_dist_init = None

    def _sigmoid_alpha(self, dist: float) -> float:
        """给定一个 dist 值，计算对应的 alpha（sigmoid 映射）"""
        alpha = self.base_alpha * (1.0 / (1.0 + np.exp(-self.sigmoid_scale * (dist - 0.5))))
        return alpha

    @torch.no_grad()
    def set_ground_truth(self, tgt_image: torch.Tensor, src_image: torch.Tensor = None):
        """
        每次调用都要重新提取目标特征。
        第一次调用时如果传了 src_image，顺便存下来并计算初始 dist（用于日志）。
        后续调用只需 tgt_image（src 已缓存）。
        """
        new_tgt_global = []
        new_tgt_local = []

        for model_idx, model in enumerate(self.extractors):
            _, all_global, all_local = model.intermediate_features(tgt_image.to(tgt_image.device))
            num_layers = len(all_global)

            while len(new_tgt_global) < num_layers:
                new_tgt_global.append({})
                new_tgt_local.append({})

            for li in range(num_layers):
                new_tgt_global[li][model_idx] = all_global[li].squeeze(0)
                new_tgt_local[li][model_idx] = all_local[li].squeeze(0)

        # 第一次调用：存源图特征 + 计算初始 dist
        # 兼容两种情况：显式传 src_image，或第一次调用时 src=None（自动用 tgt 自初始化）
        if self.dist_ema is None:
            if src_image is not None:
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
            else:
                # 未提供 src_image 时，用 tgt 自身的特征做自初始化（dist=0）
                self.src_local_per_layer = new_tgt_local

            self.gt_global_per_layer = new_tgt_global
            self.gt_local_per_layer = new_tgt_local

            # 初始化 EMA 列表（每层一个初始 dist）
            num_layers = len(new_tgt_local)
            dist_init = []
            for li in range(num_layers):
                sim_list = []
                for model_idx in new_tgt_local[li]:
                    src_p = self.src_local_per_layer[li][model_idx].unsqueeze(0)
                    tgt_p = new_tgt_local[li][model_idx].unsqueeze(0)
                    sim = F.cosine_similarity(src_p, tgt_p, dim=-1)
                    sim_list.append(sim.mean().item())
                dist_init.append(1.0 - np.mean(sim_list))

            self.dist_ema = dist_init[:]
            self.layer_dist_init = dist_init[:]

            print(f"[SemanticDistance] Init layer dist: {[f'{d:.3f}' for d in dist_init]}")
            print(f"[SemanticDistance] Init layer alphas: {[f'{self._sigmoid_alpha(d):.3f}' for d in dist_init]}")
        else:
            self.gt_global_per_layer = new_tgt_global
            self.gt_local_per_layer = new_tgt_local

    def _compute_current_dist(self, adv_local: torch.Tensor, tgt_local: torch.Tensor) -> float:
        """计算当前 adv 和 target 之间的 patch 余弦距离（单模型单层）"""
        sim = F.cosine_similarity(adv_local, tgt_local, dim=-1).mean().item()
        return 1.0 - sim

    def _compute_patch_loss(self, local_feat: torch.Tensor, tgt_local: torch.Tensor,
                            alpha: float) -> torch.Tensor:
        """Top-K 按 patch 与目标的余弦相似度降序排列：
        相似度越高 = 该 patch 已越接近目标 = 越需要被进一步强化逼近。
        切换为 largest=True 取最大相似度 patch（原来是 largest=False 取最小相似度）。
        """
        # [B, N] 沿最后一维计算与目标的余弦相似度
        sim_to_tgt = F.cosine_similarity(local_feat, tgt_local, dim=-1)  # [B, N]
        num_patches = sim_to_tgt.shape[-1]
        k = max(1, int(num_patches * self.saliency_ratio))

        # Top-K 按 patch 与目标的余弦相似度降序排列
        _, topk_idx = torch.topk(sim_to_tgt, k=k, dim=-1, largest=False)  # [B, k]

        mask = torch.zeros(sim_to_tgt.shape, dtype=torch.float32, device=sim_to_tgt.device)
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
        每步都计算当前层的实际 dist，用 EMA 平滑后算 alpha。
        """
        loss_global_total = 0.0
        loss_local_total = 0.0
        num_active = 0

        num_layers = len(self.gt_global_per_layer)

        for li in range(num_layers):
            # 获取当前层所有模型的平均当前 dist
            cur_dists = []
            for model_idx in self.gt_local_per_layer[li]:
                if model_idx in all_local_dict and li < len(all_local_dict[model_idx]):
                    adv_local = all_local_dict[model_idx][li]
                    tgt_local = self.gt_local_per_layer[li][model_idx]
                    d = self._compute_current_dist(adv_local, tgt_local)
                    cur_dists.append(d)

            if not cur_dists:
                continue

            # 当前 step 的平均 dist
            current_dist = np.mean(cur_dists)

            # EMA 平滑
            if self.dist_ema is not None and li < len(self.dist_ema):
                self.dist_ema[li] = self.ema_beta * self.dist_ema[li] + (1 - self.ema_beta) * current_dist
            else:
                # 防御性初始化
                if self.dist_ema is None:
                    self.dist_ema = [0.0] * num_layers
                while li >= len(self.dist_ema):
                    self.dist_ema.append(0.0)
                self.dist_ema[li] = current_dist

            # 用 EMA 平滑后的 dist 算 alpha
            layer_alpha = self._sigmoid_alpha(self.dist_ema[li])

            for model_idx in self.gt_global_per_layer[li]:
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
        if self.dist_ema is None:
            return [self.base_alpha] * max(len(self.gt_global_per_layer), 1)
        return [self._sigmoid_alpha(d) for d in self.dist_ema]