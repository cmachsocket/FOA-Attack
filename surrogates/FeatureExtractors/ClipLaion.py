import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms


def gram_matrix(features):
    """Compute Gram matrix from spatial features [B, C, H, W]."""
    B, C, H, W = features.shape
    features_flat = features.view(B, C, H * W)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
    return gram / (C * H * W)


class ClipLaionFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s34b-b79k").cuda()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        return global_feature, local_feature

    def spatial_gram_features(self, x):
        """Return spatial [B,C,H,W] patch features reshaped from patch tokens + Gram matrix."""
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
        features = outputs.last_hidden_state          # [B, 257, 1408] (1 CLS + 256 patches)
        patch_features = features[:, 1:, :]          # [B, 256, 1408]
        B, N, D = patch_features.shape                 # B=1, N=256, D=1408
        H = W = int(N ** 0.5)                          # 16
        spatial = patch_features.transpose(1, 2).view(B, D, H, W)  # [B, D, 16, 16]
        gram = gram_matrix(spatial)
        return spatial, gram


class ClipLaionFeatureExtractorOT(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionFeatureExtractorOT, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s34b-b79k").cuda()
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_embedding(inputs["pixel_values"])
        global_feature = features[:,0,:]
        local_feature = features[:,1:,:]
        return global_feature, local_feature