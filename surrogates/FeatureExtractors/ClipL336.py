import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms


class ClipL336FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipL336FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").cuda()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(336),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
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

    def intermediate_features(self, x):
        """Return features from all transformer layers."""
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True,
        )
        all_hidden_states = outputs.hidden_states

        all_features, all_global, all_local = [], [], []
        for hs in all_hidden_states:
            all_features.append(hs)
            cls_tok = hs[:, 0, :] / (hs[:, 0, :].norm(dim=1, keepdim=True) + 1e-8)
            patch_tok = hs[:, 1:, :] / (hs[:, 1:, :].norm(dim=1, keepdim=True) + 1e-8)
            all_global.append(cls_tok)
            all_local.append(patch_tok)

        return all_features, all_global, all_local
