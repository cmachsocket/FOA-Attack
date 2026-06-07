"""
InternVL3-1B Feature Extractor

跟 Qwen2.5-VL 同源（都来自 InternVL 体系），作为 surrogate 可以
让 CLIP-only ensemble 多出一个"接近 victim vision encoder"的视角，
提升迁移攻击到 VLM 上的效果。

模型：OpenGVLab/InternVL3-1B
- Vision tower: InternViT-300M (24 layers, hidden=1024, 16 heads)
- LLM: Qwen2.5-1.8B（不需要，提取后立即释放）
- 输入: 448x448, ImageNet normalization
- Patch: 14x14, 32x32 = 1024 patches + 1 CLS = 1025 tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from .Base import BaseFeatureExtractor
from torchvision import transforms

# ---- monkey-patch transformers 5.x incompatibility: InternVL3-1B uses
# a model whose language_model has no `all_tied_weights_keys` (only
# `_tied_weights_keys`).  This is a bug in transformers 5.9+ that appeared
# when InternVL3 was last updated.  Patch it before loading the model.
_orig_getattr = nn.Module.__getattr__
def _patched_nn_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
nn.Module.__getattr__ = _patched_nn_getattr


INTERNVL_MEAN = (0.485, 0.456, 0.406)
INTERNVL_STD = (0.229, 0.224, 0.225)
INTERNVL_IMG_SIZE = 448
INTERNVL_PATCH_SIZE = 14
INTERNVL_NUM_PATCHES = (INTERNVL_IMG_SIZE // INTERNVL_PATCH_SIZE) ** 2  # 1024


class InternVL3FeatureExtractor(BaseFeatureExtractor):
    """
    提取 InternViT 的 features。

    只用 vision tower，不加载 LLM，加载完立即释放以节省 VRAM。
    bfloat16 推理，跟原训练一致。
    """

    def __init__(self, model_path: str = "OpenGVLab/InternVL3-1B", device: str = "cuda"):
        super(InternVL3FeatureExtractor, self).__init__()
        self.device_target = device

        # 加载完整 chat model（trust_remote_code 必须）
        # generation_config=None: 避免 transformers 5.x 尝试加载 hub 上的 custom_generate/generate.py
        # （该文件在 InternVL3-1B 的 hub 上不存在，会报网络错误）
        full_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            generation_config=None,
        )

        # 只保留 vision tower
        self.vision_model = full_model.vision_model
        self.vision_model = self.vision_model.eval().to(device)

        # 显式释放 LLM 和 projector
        del full_model.language_model
        del full_model.mlp1
        del full_model
        torch.cuda.empty_cache()

        # 跟 CLIP normalizer 同一套 interface：input ∈ [0,255] → [0,1] → normalize
        self.normalizer = transforms.Compose([
            transforms.Resize(
                INTERNVL_IMG_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(INTERNVL_IMG_SIZE),
            transforms.Normalize(INTERNVL_MEAN, INTERNVL_STD),
        ])

        # hidden_size for downstream losses that need D
        self.hidden_size = self.vision_model.config.hidden_size  # 1024
        self.num_patches = INTERNVL_NUM_PATCHES  # 1024

    def _encode(self, x: torch.Tensor, output_hidden_states: bool = False):
        """
        跑 vision tower。注意：不加 @torch.no_grad()，因为
        ensemble attack 需要对这个 forward 算 grad，backprop 到输入 x。
        模型本身在 get_models_ot 里用 .requires_grad_(False) 冻住，权重
        不会更新，只有输入 delta 会有梯度。

        Args:
            x: [B, 3, H, W] in [0, 255]
            output_hidden_states: 是否返回所有层 hidden_states
        Returns:
            last_hidden_state: [B, 1025, 1024]
            pooled_output (CLS): [B, 1024]
            (optional) all_hidden_states: tuple of [B, 1025, 1024]
        """
        pixel_values = self.normalizer(x).to(self.device_target, dtype=torch.bfloat16)
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B, 1024] CLS pooled feature（L2-normed）"""
        out = self._encode(x, output_hidden_states=False)
        cls = out.pooler_output.float()
        return cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)

    def global_local_features(self, x: torch.Tensor):
        """
        跟 CLIP 接口对齐。
        Returns:
            global_feature: [B, 1024] L2-normed CLS
            local_feature:  [B, 1024, 1024] L2-normed patch tokens
        """
        out = self._encode(x, output_hidden_states=False)
        hs = out.last_hidden_state.float()  # [B, 1025, 1024]
        cls = hs[:, 0, :]
        patches = hs[:, 1:, :]
        cls = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
        patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)
        return cls, patches

    def intermediate_features(self, x: torch.Tensor):
        """
        跟 CLIP 接口对齐，返回所有层（embeddings + 24 layers = 25 个 hidden states）。

        Returns:
            all_features:  list of [B, 1025, 1024] per layer (含 input proj 输出)
            all_global:    list of [B, 1024] CLS tokens per layer (L2-normed)
            all_local:     list of [B, 1024, 1024] patch tokens per layer (L2-normed)
        """
        out = self._encode(x, output_hidden_states=True)
        all_hidden_states = out.hidden_states  # tuple of (embeddings_out + 24 layers) = 25

        all_features, all_global, all_local = [], [], []
        for hs in all_hidden_states:
            hs_f = hs.float()
            all_features.append(hs_f)
            cls = hs_f[:, 0, :]
            patches = hs_f[:, 1:, :]
            cls = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
            patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)
            all_global.append(cls)
            all_local.append(patches)

        return all_features, all_global, all_local

    def spatial_gram_features(self, x: torch.Tensor):
        """
        [B, 1024, 32, 32] 空间特征 + Gram matrix（用 raw，未 L2-norm）。
        """
        out = self._encode(x, output_hidden_states=False)
        patch_features = out.last_hidden_state[:, 1:, :].float()  # [B, 1024, 1024]
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)  # 32
        spatial = patch_features.transpose(1, 2).view(B, D, H, W)

        # Gram（局部实现，避免依赖 ClipB16 的同名函数）
        B_, C, Hh, Ww = spatial.shape
        flat = spatial.view(B_, C, Hh * Ww)
        gram = torch.bmm(flat, flat.transpose(1, 2)) / (C * Hh * Ww)
        return spatial, gram


def gram_matrix(features):
    """Compute Gram matrix from spatial features [B, C, H, W]."""
    B, C, H, W = features.shape
    features_flat = features.view(B, C, H * W)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
    return gram / (C * H * W)
