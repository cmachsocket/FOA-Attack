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
    def set_ground_truth(self, x: torch.Tensor):
        """设置目标图像的特征"""
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            self.ground_truth.append(x_tensor.squeeze(0))
            # 存储原始 patch embeddings（不聚类），用于 saliency loss
            self.ground_truth_local.append(x_embedding.squeeze(0))  # [N_patches, D]
    
    def compute_saliency_mask(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算显著性 mask
        
        Args:
            features: [N, D] local features (N = num patches)
        Returns:
            mask: [N] 1.0 for high saliency, 0.0 for others
        """
        # 基于特征范数作为显著性指标
        feat_norm = torch.norm(features, dim=-1)  # [N]
        
        # 取 top-k 高显著性的 patch
        k = int(features.shape[0] * self.saliency_ratio)
        k = max(1, k)  # 至少选1个
        
        _, topk_idx = torch.topk(feat_norm, k=k, dim=-1)
        
        mask = torch.zeros_like(feat_norm)
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
            # 计算源图像的显著性 mask
            saliency_mask = self.compute_saliency_mask(local_feat)  # [N]
            
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
    def set_ground_truth(self, x: torch.Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            self.ground_truth.append(x_tensor.squeeze(0))
            # 存储原始 patch embeddings（不聚类）
            self.ground_truth_local.append(x_embedding.squeeze(0))
    
    def compute_attention_saliency(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        基于注意力机制计算显著性
        
        使用 query-key 相似度作为注意力权重
        """
        # 使用均值池化作为 query
        query = local_feat.mean(dim=0, keepdim=True)  # [1, D]
        
        # 计算每个 patch 与 query 的相似度
        attn_scores = torch.sum(local_feat * query, dim=-1)  # [N]
        
        return attn_scores
    
    def compute_saliency_mask(self, features: torch.Tensor) -> torch.Tensor:
        """组合范数和注意力计算显著性"""
        # 范数显著性
        norm_saliency = torch.norm(features, dim=-1)
        norm_saliency = (norm_saliency - norm_saliency.min()) / (norm_saliency.max() - norm_saliency.min() + 1e-8)
        
        # 注意力显著性
        attn_saliency = self.compute_attention_saliency(features)
        attn_saliency = (attn_saliency - attn_saliency.min()) / (attn_saliency.max() - attn_saliency.min() + 1e-8)
        
        # 组合
        combined_saliency = 0.5 * norm_saliency + 0.5 * attn_saliency
        
        # 取 top-k
        k = int(features.shape[0] * self.saliency_ratio)
        k = max(1, k)
        _, topk_idx = torch.topk(combined_saliency, k=k, dim=-1)
        
        mask = torch.zeros_like(combined_saliency)
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
            
            # 显著性 mask
            saliency_mask = self.compute_saliency_mask(local_feat)
            
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
                 high_alpha: float = 0.9,
                 mid_alpha: float = 0.5):
        super(SaliencySuppressionReconstructionLossV3, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.high_ratio = high_ratio      # top 10%
        self.mid_ratio = mid_ratio         # 10%-30%
        self.high_alpha = high_alpha       # 强抑制
        self.mid_alpha = mid_alpha         # 中等抑制
        self.step_count = 0
    
    @torch.no_grad()
    def set_ground_truth(self, x: torch.Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            self.ground_truth.append(x_tensor.squeeze(0))
            # 存储原始 patch embeddings（不聚类）
            self.ground_truth_local.append(x_embedding.squeeze(0))
    
    def compute_multi_level_saliency_mask(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算多层级显著性 mask"""
        feat_norm = torch.norm(features, dim=-1)
        
        # 排序获取阈值
        sorted_norm, _ = torch.sort(feat_norm, descending=True)
        n = features.shape[0]
        
        high_k = int(n * self.high_ratio)
        mid_k = int(n * (self.high_ratio + self.mid_ratio))
        
        # 多层级 mask
        high_mask = torch.zeros_like(feat_norm)
        mid_mask = torch.zeros_like(feat_norm)
        low_mask = torch.zeros_like(feat_norm)
        
        if high_k > 0:
            high_mask[:high_k] = 1.0
        if mid_k > high_k:
            mid_mask[high_k:mid_k] = 1.0
        if mid_k < n:
            low_mask[mid_k:] = 1.0
        
        return {"high": high_mask, "mid": mid_mask, "low": low_mask}
    
    def get_alpha_schedule(self) -> Dict[str, float]:
        """渐进式 alpha 调度"""
        progress = min(self.step_count / 300, 1.0)
        
        # 不同层级的 alpha 都递增，但速度不同
        base_alpha = 0.1 + 0.9 * (1 - np.cos(np.pi * progress)) / 2
        
        return {
            "high": min(base_alpha * 1.2, 1.0),
            "mid": base_alpha,
            "low": base_alpha * 0.5
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
            
            # 多层级显著性 mask
            masks = self.compute_multi_level_saliency_mask(local_feat)
            
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