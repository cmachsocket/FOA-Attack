"""
FOA-Saliency Attack: Saliency-based Suppression and Reconstruction for Adversarial Transferability

核心思路：
1. 提取 source 和 target 的 local features (patch tokens)
2. 计算每个 patch 的显著性（特征范数）
3. 对高显著性区域进行渐进式抑制
4. 在同一位置用 target 的 patch 重建
5. 损失：重建后的特征 vs target 特征（余弦相似度）

去掉 OT（最优传输），改用空间位置直接对应的方式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class SaliencySuppressionReconstructionLoss(nn.Module):
    """
    显著性抑制 + 重建损失
    
    对高显著性的 patch 进行抑制，然后在同一位置用 target 的 patch 重建。
    通过余弦相似度约束空间对齐，不再使用 OT 匹配。
    """
    
    def __init__(self, extractors: List[nn.Module], saliency_ratio: float = 0.3, alpha_schedule: str = "linear"):
        super(SaliencySuppressionReconstructionLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.saliency_ratio = saliency_ratio  # 抑制前 X% 高显著区域
        self.alpha_schedule = alpha_schedule  # "linear" or "cosine"
        self.step_count = 0
    
    @torch.no_grad()
    def set_ground_truth(self, tgt: torch.Tensor, src: torch.Tensor = None):
        """设置目标图像的特征，src 参数仅兼容接口，实际不使用"""
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            tgt_tensor, tgt_embedding = model.global_local_features(tgt.to(tgt.device))
            self.ground_truth.append(tgt_tensor.squeeze(0))
            self.ground_truth_local.append(tgt_embedding.squeeze(0))  # [N_patches, D]
    
    def compute_saliency_mask(self, local_feat: torch.Tensor, tgt_local: torch.Tensor) -> torch.Tensor:
        """
        基于当前 adv patch 与目标 patch 的余弦相似度计算显著性 mask
        选最不相似的 patch（largest=False），即距离目标最远的区域优先抑制
        
        Args:
            local_feat: [N, D] 当前对抗图的 patch features
            tgt_local: [N, D] 目标图的 patch features
        Returns:
            mask: [N] 1.0 for high saliency (most different from target), 0.0 for others
        """
        # 计算每个 patch 与目标 patch 的余弦相似度
        sim_to_tgt = F.cosine_similarity(local_feat, tgt_local, dim=-1)  # [N]
        # 相似度越低 = 离目标越远 = 越需要抑制
        k = int(local_feat.shape[0] * self.saliency_ratio)
        k = max(1, k)
        
        _, topk_idx = torch.topk(sim_to_tgt, k=k, dim=-1, largest=False)
        
        mask = torch.zeros_like(sim_to_tgt)
        mask.scatter_(0, topk_idx, 1.0)
        
        return mask
    
    def get_current_alpha(self, total_steps: int) -> float:
        """
        获取当前迭代的 alpha（抑制强度）
        
        渐进式策略：alpha 从小到大逐渐增加
        """
        if self.alpha_schedule == "linear":
            # 线性递增：0.1 -> 1.0
            progress = min(self.step_count / total_steps, 1.0)
            alpha = 0.1 + 0.9 * progress
        elif self.alpha_schedule == "cosine":
            # 余弦递增：前期慢，后期快
            progress = min(self.step_count / total_steps, 1.0)
            alpha = 0.1 + 0.9 * (1 - np.cos(np.pi * progress)) / 2
        elif self.alpha_schedule == "step":
            # 阶梯式：每隔一定步数跳变
            stage = self.step_count // (total_steps // 3)
            alpha = [0.1, 0.5, 0.8, 1.0][min(stage, 3)]
        else:
            alpha = 0.7
        
        return alpha
    
    def forward(self, feature_dict: Dict[int, torch.Tensor], 
               feature_local_dict: Dict[int, torch.Tensor], 
               total_steps: int = 300,
               y: any = None) -> torch.Tensor:
        """
        计算显著性抑制重建损失
        
        Args:
            feature_dict: 全局特征字典 {model_idx: features}
            feature_local_dict: 局部特征字典 {model_idx: local_features}
            total_steps: 总迭代步数（用于 alpha 调度）
            y: 未使用，兼容接口
        """
        self.step_count += 1
        alpha = self.get_current_alpha(total_steps)
        
        loss_global = 0
        loss_local = 0
        
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            gt_local = self.ground_truth_local[index]
            
            feature = feature_dict[index].unsqueeze(0)  # [1, D]
            local_feat = feature_local_dict[index]       # [N, D]
            
            # ========== 全局损失（CLS token 对齐）==========
            # 保持 FOA 的全局余弦相似度损失
            feat_loss = torch.sum(feature * gt, dim=-1).mean()
            
            # ========== 局部损失（显著性抑制 + 重建）==========
            # 使用当前 adv patch vs 目标 patch 的相似度计算显著性 mask
            saliency_mask = self.compute_saliency_mask(local_feat, gt_local)  # [N]
            
            # 抑制高显著性区域 + 用目标对应位置重建
            # src' = src * (1 - alpha * mask) + tgt * (alpha * mask)
            reconstructed = local_feat * (1 - alpha * saliency_mask.unsqueeze(-1)) \
                          + gt_local * (alpha * saliency_mask.unsqueeze(-1))
            
            # 同一空间位置直接计算余弦相似度损失（不用 OT）
            # 只对高显著性区域计算损失
            sim = F.cosine_similarity(reconstructed, gt_local, dim=-1)  # [N]
            
            # 加权：只对高显著性区域计算损失
            local_loss = (1 - sim) * saliency_mask  # 高显著区域 loss
            local_loss = local_loss.sum() / (saliency_mask.sum() + 1e-8)
            
            loss_global += feat_loss
            loss_local += local_loss
        
        # 归一化
        loss_global = loss_global / len(self.extractors)
        loss_local = loss_local / len(self.extractors)
        
        # 组合：全局 + 局部（局部权重 0.2）
        total_loss = loss_global + 0.2 * loss_local

        # 确保返回类型为 torch.Tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, dtype=torch.float32, device=(loss_global.device if isinstance(loss_global, torch.Tensor) else None))

        return total_loss


class SaliencySuppressionReconstructionLossV2(nn.Module):
    """
    改进版：加入空间注意力机制
    
    特点：
    1. 使用注意力图计算显著性（不只是范数）
    2. 考虑空间邻近关系
    3. 引入空间一致性正则化
    """
    
    def __init__(self, extractors: List[nn.Module], saliency_ratio: float = 0.3):
        super(SaliencySuppressionReconstructionLossV2, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.saliency_ratio = saliency_ratio
        self.step_count = 0
    
    @torch.no_grad()
    def set_ground_truth(self, tgt: torch.Tensor, src: torch.Tensor = None):
        """设置目标图像的特征，src 参数仅兼容接口"""
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            tgt_tensor, tgt_embedding = model.global_local_features(tgt.to(tgt.device))
            self.ground_truth.append(tgt_tensor.squeeze(0))
            self.ground_truth_local.append(tgt_embedding.squeeze(0))
    
    def compute_saliency_mask(self, local_feat: torch.Tensor, tgt_local: torch.Tensor) -> torch.Tensor:
        """
        基于当前 adv patch 与目标 patch 的余弦相似度计算显著性 mask
        选最不相似的 patch（largest=False），即距离目标最远的区域优先抑制
        """
        sim_to_tgt = F.cosine_similarity(local_feat, tgt_local, dim=-1)  # [N]
        k = int(local_feat.shape[0] * self.saliency_ratio)
        k = max(1, k)
        _, topk_idx = torch.topk(sim_to_tgt, k=k, dim=-1, largest=False)
        
        mask = torch.zeros_like(sim_to_tgt)
        mask.scatter_(0, topk_idx, 1.0)
        
        return mask
    
    def get_alpha(self) -> float:
        """渐进式 alpha"""
        progress = min(self.step_count / 300, 1.0)
        return 0.1 + 0.9 * (1 - np.cos(np.pi * progress)) / 2
    
    def forward(self, feature_dict: Dict[int, torch.Tensor], 
               feature_local_dict: Dict[int, torch.Tensor], 
               total_steps: int = 300,
               y: any = None) -> torch.Tensor:
        self.step_count += 1
        alpha = self.get_alpha()
        
        loss_global = 0
        loss_local = 0
        
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            gt_local = self.ground_truth_local[index]
            
            feature = feature_dict[index].unsqueeze(0)
            local_feat = feature_local_dict[index]
            
            # 全局损失
            feat_loss = torch.sum(feature * gt, dim=-1).mean()
            
            # 显著性 mask（使用当前 adv patch 计算）
            saliency_mask = self.compute_saliency_mask(local_feat, gt_local)
            
            # 抑制 + 重建
            reconstructed = local_feat * (1 - alpha * saliency_mask.unsqueeze(-1)) \
                          + gt_local * (alpha * saliency_mask.unsqueeze(-1))
            
            # 位置对应余弦损失
            sim = F.cosine_similarity(reconstructed, gt_local, dim=-1)
            local_loss = ((1 - sim) * saliency_mask).sum() / (saliency_mask.sum() + 1e-8)
            
            loss_global += feat_loss
            loss_local += local_loss
        
        loss_global /= len(self.extractors)
        loss_local /= len(self.extractors)

        total = loss_global + 0.2 * loss_local
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(total, dtype=torch.float32, device=(loss_global.device if isinstance(loss_global, torch.Tensor) else None))

        return total


class SaliencySuppressionReconstructionLossV3(nn.Module):
    """
    进阶版：多层渐进抑制
    
    对不同显著层级的区域使用不同的抑制策略：
    - 超高显著（top 10%）：强抑制
    - 高显著（10%-30%）：中等抑制
    - 低显著（30%-）：弱或无抑制
    """
    
    def __init__(self, extractors: List[nn.Module], 
                 high_ratio: float = 0.1, 
                 mid_ratio: float = 0.2,
                 high_alpha: float = 1.2,
                 mid_alpha: float = 1.0,
                 low_alpha: float = 0.5):
        super(SaliencySuppressionReconstructionLossV3, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.high_ratio = high_ratio      # top 10%
        self.mid_ratio = mid_ratio         # 10%-30%
        self.high_alpha = high_alpha       # 高显著倍率（× base_alpha）
        self.mid_alpha = mid_alpha         # 中显著倍率（× base_alpha）
        self.low_alpha = low_alpha         # 低显著倍率（× base_alpha）
        self.step_count = 0
    
    @torch.no_grad()
    def set_ground_truth(self, tgt: torch.Tensor, src: torch.Tensor = None):
        """设置目标图像的特征，src 参数仅兼容接口"""
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            tgt_tensor, tgt_embedding = model.global_local_features(tgt.to(tgt.device))
            self.ground_truth.append(tgt_tensor.squeeze(0))
            self.ground_truth_local.append(tgt_embedding.squeeze(0))
    
    def compute_multi_level_saliency_mask(self, local_feat: torch.Tensor, tgt_local: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        基于与目标 patch 的余弦相似度计算多层级显著性 mask
        相似度越低 = 离目标越远 = 优先级越高
        """
        sim_to_tgt = F.cosine_similarity(local_feat, tgt_local, dim=-1)  # [N]
        n = local_feat.shape[0]
        
        # 按相似度升序排序（最不像目标的排前面）
        sorted_sim, sorted_idx = torch.sort(sim_to_tgt, descending=False)
        
        high_k = int(n * self.high_ratio)
        mid_k = int(n * (self.high_ratio + self.mid_ratio))
        
        # 多层级 mask（基于排序索引）
        high_mask = torch.zeros_like(sim_to_tgt)
        mid_mask = torch.zeros_like(sim_to_tgt)
        low_mask = torch.zeros_like(sim_to_tgt)
        
        if high_k > 0:
            high_mask[sorted_idx[:high_k]] = 1.0
        if mid_k > high_k:
            mid_mask[sorted_idx[high_k:mid_k]] = 1.0
        if mid_k < n:
            low_mask[sorted_idx[mid_k:]] = 1.0
        
        return {"high": high_mask, "mid": mid_mask, "low": low_mask}
    
    def get_alpha_schedule(self) -> Dict[str, float]:
        """渐进式 alpha 调度：各层 = base_alpha × 该层倍率"""
        progress = min(self.step_count / 300, 1.0)
        
        # base_alpha 按 step 走 cosine 曲线从 0.1 递增到 1.0
        base_alpha = 0.1 + 0.9 * (1 - np.cos(np.pi * progress)) / 2
        
        return {
            "high": min(base_alpha * self.high_alpha, 1.0),
            "mid": min(base_alpha * self.mid_alpha, 1.0),
            "low": min(base_alpha * self.low_alpha, 1.0)
        }
    
    def forward(self, feature_dict: Dict[int, torch.Tensor], 
               feature_local_dict: Dict[int, torch.Tensor], 
               total_steps: int = 300,
               y: any = None) -> torch.Tensor:
        self.step_count += 1
        alpha_schedule = self.get_alpha_schedule()
        
        loss_global = 0
        loss_local = 0
        
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            gt_local = self.ground_truth_local[index]
            
            feature = feature_dict[index].unsqueeze(0)
            local_feat = feature_local_dict[index]
            
            # 全局损失
            feat_loss = torch.sum(feature * gt, dim=-1).mean()
            
            # 多层级显著性 mask（使用当前 adv patch 计算）
            masks = self.compute_multi_level_saliency_mask(local_feat, gt_local)
            
            # 分层抑制 + 重建
            reconstructed = local_feat.clone()
            for level, mask in masks.items():
                if mask.sum() > 0:
                    alpha = alpha_schedule[level]
                    # 使用该层级的 alpha 进行抑制
                    suppression = mask.unsqueeze(-1) * alpha
                    reconstructed = reconstructed * (1 - suppression) + gt_local * suppression
            
            # 计算损失（只对高显著区域）
            sim = F.cosine_similarity(reconstructed, gt_local, dim=-1)
            high_mask = masks["high"]
            local_loss = ((1 - sim) * high_mask).sum() / (high_mask.sum() + 1e-8)
            
            loss_global += feat_loss
            loss_local += local_loss
        
        loss_global /= len(self.extractors)
        loss_local /= len(self.extractors)

        total = loss_global + 0.2 * loss_local
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(total, dtype=torch.float32, device=(loss_global.device if isinstance(loss_global, torch.Tensor) else None))

        return total