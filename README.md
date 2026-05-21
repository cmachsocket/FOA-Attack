# FOA-Saliency Attack

基于显著性抑制与重建的对抗攻击方法，去掉了 FOA-Attack 中的最优传输（OT）匹配，改用空间位置直接对应的策略。

## 核心思想

```
源图像 Patch Tokens → 高显著性区域抑制 → 同一位置用目标图像 Patch 重建
                         ↓
                    余弦相似度损失（不用 OT）
```

## 与 FOA-Attack 的区别

| 方面 | FOA-Attack | Saliency Attack |
|------|-----------|-----------------|
| 局部对齐 | K-means 聚类 + OT Sinkhorn | 空间位置直接对应 |
| 高显著性 | 隐式 | 显式（特征范数/注意力） |
| 匹配方式 | 最优传输（OT） | 无需匹配 |
| 计算开销 | 高（Sinkhorn 迭代） | 低（直接余弦） |

## 文件结构

```
FOA-Attack/
├── saliency_loss.py          # 核心损失函数（3 个版本）
├── generate_adversarial_samples_saliency.py  # 入口脚本
├── config/
│   └── saliency_attack.yaml  # 配置文件
└── README.md
```

## 三个损失版本

### V1: 基础版
- 基于特征范数计算显著性
- 线性/余弦递增的 alpha 调度
- 单一层级的抑制

### V2: 注意力版
- 组合特征范数 + 注意力分数
- 更准确的显著性估计

### V3: 多层渐进版
- 超高显著（top 10%）：强抑制
- 高显著（10%-30%）：中等抑制
- 低显著（30%+）：弱或无抑制

## 使用方法

```bash
# V1 基础版
python generate_adversarial_samples_saliency.py \
    model.saliency_loss_version=v1 \
    model.saliency_ratio=0.3

# V2 注意力版
python generate_adversarial_samples_saliency.py \
    model.saliency_loss_version=v2 \
    model.saliency_ratio=0.3

# V3 多层渐进版
python generate_adversarial_samples_saliency.py \
    model.saliency_loss_version=v3 \
    model.high_ratio=0.1 \
    model.mid_ratio=0.2
```

## 核心公式

### 显著性 mask 计算
```
saliency_mask = top_k(normalize(||patch_feat||_2), k)
```

### 抑制 + 重建
```
reconstructed_patch = src_patch * (1 - α * mask) + tgt_patch * (α * mask)
```

### 损失函数
```
L = cos(reconstructed_patch, tgt_patch)  # 对高显著区域
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `saliency_ratio` | 抑制前 X% 高显著区域 | 0.3 |
| `alpha_schedule` | α 调度策略 | "cosine" |
| `saliency_loss_version` | 损失版本 | "v1" |

## 实验建议

1. **先跑 V1**：验证基础思路有效性
2. **再跑 V2**：看注意力机制是否提升
3. **最后跑 V3**：多层渐进可能效果最好，但需要调参
4. **调参重点**：`saliency_ratio`（0.1~0.5）、`alpha_schedule`（linear/cosine/step）