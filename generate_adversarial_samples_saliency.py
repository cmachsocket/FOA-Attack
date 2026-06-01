"""
FOA-Saliency Attack 入口脚本

使用方法：
    python generate_adversarial_samples_saliency.py

替换原有的 generate_adversarial_samples_foa_attack.py 中的 OT 损失，
改用显著性抑制 + 重建策略。
"""

import os
import json
import hashlib
import random
import argparse
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image


def _patch_argparse_help_for_hydra_py314() -> None:
    """Compat patch for hydra-core 1.3.x on Python 3.14 argparse help checks."""
    formatter_cls = argparse.HelpFormatter
    if getattr(formatter_cls, "_hydra_py314_help_patch", False):
        return

    original_expand_help = formatter_cls._expand_help

    def _expand_help_compat(self, action):
        help_obj = action.help
        if help_obj is not None and not isinstance(help_obj, str):
            try:
                action.help = str(help_obj)
                return original_expand_help(self, action)
            finally:
                action.help = help_obj
        return original_expand_help(self, action)

    formatter_cls._expand_help = _expand_help_compat
    formatter_cls._hydra_py314_help_patch = True


_patch_argparse_help_for_hydra_py314()

import hydra
from omegaconf import DictConfig
import os
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from pytorch_lightning import seed_everything
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
    EnsembleFeatureExtractor_ot,
)

from saliency_loss import (
    SaliencySuppressionReconstructionLoss,
    SaliencySuppressionReconstructionLossV2,
    SaliencySuppressionReconstructionLossV3,
)
from semantic_distance_loss import SemanticDistanceLoss

from utils import hash_training_config, setup_wandb, ensure_dir

# Mapping from backbone names to model classes
BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}

# 损失版本映射
SALIENCY_LOSS_MAP = {
    "v1": SaliencySuppressionReconstructionLoss,
    "v2": SaliencySuppressionReconstructionLossV2,
    "v3": SaliencySuppressionReconstructionLossV3,
}


def get_models_ot(cfg: MainConfig):
    """Get models based on configuration."""
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=10)
    else:
        ensemble_extractor = models[0]

    return ensemble_extractor, models


def get_saliency_loss(cfg: MainConfig, models: List[nn.Module], version: str = "v1"):
    """
    获取显著性抑制重建损失
    
    Args:
        cfg: 配置
        models: 模型列表
        version: 损失版本 ("v1", "v2", "v3", "semantic_distance")
    """
    saliency_ratio = getattr(cfg.model, 'saliency_ratio', 0.3)
    
    if version == "semantic_distance":
        return SemanticDistanceLoss(
            extractors=models,
            saliency_ratio=saliency_ratio,
            base_alpha=getattr(cfg.model, 'base_alpha', 0.5),
            sigmoid_scale=getattr(cfg.model, 'sigmoid_scale', 6.0),
            local_weight=0.2,
            ema_beta=getattr(cfg.model, 'ema_beta', 0.9),
        )
    
    loss_class = SALIENCY_LOSS_MAP.get(version, SaliencySuppressionReconstructionLoss)
    
    saliency_loss = loss_class(
        extractors=models,
        saliency_ratio=saliency_ratio
    )
    
    return saliency_loss


def set_environment(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    """Convert PIL.Image to PyTorch Tensor"""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@hydra.main(version_base=None, config_path="config", config_name="saliency_attack")
def main(cfg: MainConfig):
    set_environment()

    # 初始化 wandb
    setup_wandb(cfg, tags=["saliency_attack"])
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # 获取模型
    ensemble_extractor_ot, models = get_models_ot(cfg)
    
    # 获取显著性损失（替换 OT 损失）
    saliency_version = getattr(cfg.model, 'saliency_loss_version', 'v1')
    saliency_loss = get_saliency_loss(cfg, models, version=saliency_version)

    print(f"Using Saliency Loss Version: {saliency_version}")
    print(f"Saliency Ratio: {getattr(cfg.model, 'saliency_ratio', 0.3)}")

    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    print("Using source crop:", cfg.model.use_source_crop)
    print("Using target crop:", cfg.model.use_target_crop)

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )

    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt)) in enumerate(
        zip(data_loader_imagenet, data_loader_target)
    ):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        # 跳过已生成的样本
        config_hash = hash_training_config(cfg)
        first_name = path_org[0].split('/')[-1]
        save_name = first_name[:-4] + "png" if "JPEG" in first_name else first_name
        folder_check = os.path.join(cfg.data.output, "img", config_hash, path_org[0].split('/')[-2])
        if os.path.exists(os.path.join(folder_check, save_name)):
            print(f"  ⏭ skip {save_name}, already exists")
            continue

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size} | target: {path_tgt[0].split('/')[-2]}/{path_tgt[0].split('/')[-1]}")

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor_ot,
            saliency_loss=saliency_loss,
            source_crop=source_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            path_tgt=path_tgt,
            image_tgt=image_tgt,
            target_crop=target_crop,
        )

    wandb.finish()


def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    saliency_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    path_tgt: List[str],
    image_tgt: torch.Tensor,
):
    print(f"  source: {path_org[0].split('/')[-2]}/{path_org[0].split('/')[-1]}  →  target: {path_tgt[0].split('/')[-2]}/{path_tgt[0].split('/')[-1]}")
    image_org, image_tgt = image_org.to(cfg.model.device), image_tgt.to(cfg.model.device)
    attack_type = cfg.attack.type
    attack_fn = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }[attack_type]
    saliency_version = getattr(cfg.model, 'saliency_loss_version', 'v1')
    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        saliency_loss=saliency_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
    )

    config_hash = hash_training_config(cfg)

    for path_idx in range(len(path_org)):
        folder, name = (
            path_org[path_idx].split("/")[-2],
            path_org[path_idx].split("/")[-1],
        )
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        if "JPEG" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + "png"
            )
        elif "png" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name)
            )


def log_metrics(pbar, metrics, img_index, epoch=None):
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    wandb_metrics = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch

    wandb.log(wandb_metrics)


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    saliency_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """FGSM attack with Saliency Suppression Reconstruction Loss"""
    saliency_version = getattr(cfg.model, 'saliency_loss_version', 'v1')
    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Saliency Attack progress")
    total_steps = cfg.optim.steps

    for epoch in pbar:
        with torch.no_grad():
            if saliency_version == "semantic_distance":
                saliency_loss.set_ground_truth(target_crop(image_tgt), src_image=image_org)
            else:
                saliency_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # 使用显著性损失（传入原始 patch embeddings）
        if saliency_version == "semantic_distance":
            all_global, all_local = ensemble_extractor.intermediate_forward(adv_image)
            global_sim = saliency_loss(all_global, all_local, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                all_global_crop, all_local_crop = ensemble_extractor.intermediate_forward(local_cropped)
                local_sim = saliency_loss(all_global_crop, all_local_crop, total_steps=total_steps)
                loss = local_sim
                metrics["local_similarity"] = local_sim.item()
            else:
                loss = global_sim
        else:
            adv_features, adv_features_local, adv_features_raw = ensemble_extractor(adv_image)
            global_sim = saliency_loss(adv_features, adv_features_raw, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                local_features, local_features_local, local_features_raw = ensemble_extractor(local_cropped)
                if local_features_raw and sum(v.abs().sum() for v in local_features_raw.values()) > 0:
                    local_sim = saliency_loss(local_features, local_features_raw, total_steps=total_steps)
                    loss = local_sim
                    metrics["local_similarity"] = local_sim.item()
                else:
                    loss = global_sim
                    metrics["local_similarity"] = float("nan")
            else:
                loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
    
    return adv_image


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    saliency_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """MI-FGSM attack with Saliency Suppression Reconstruction Loss"""
    saliency_version = getattr(cfg.model, 'saliency_loss_version', 'v1')
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org, requires_grad=False)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Saliency Attack progress")
    total_steps = cfg.optim.steps

    for epoch in pbar:
        with torch.no_grad():
            if saliency_version == "semantic_distance":
                saliency_loss.set_ground_truth(target_crop(image_tgt), src_image=image_org)
            else:
                saliency_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        if saliency_version == "semantic_distance":
            all_global, all_local = ensemble_extractor.intermediate_forward(adv_image)
            global_sim = saliency_loss(all_global, all_local, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                all_global_crop, all_local_crop = ensemble_extractor.intermediate_forward(local_cropped)
                local_sim = saliency_loss(all_global_crop, all_local_crop, total_steps=total_steps)
                loss = local_sim
                metrics["local_similarity"] = local_sim.item()
            else:
                loss = global_sim
        else:
            adv_features, adv_features_local, adv_features_raw = ensemble_extractor(adv_image)
            global_sim = saliency_loss(adv_features, adv_features_raw, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                local_features, local_features_local, local_features_raw = ensemble_extractor(local_cropped)
                if local_features_raw and sum(v.abs().sum() for v in local_features_raw.values()) > 0:
                    local_sim = saliency_loss(local_features, local_features_raw, total_steps=total_steps)
                    loss = local_sim
                    metrics["local_similarity"] = local_sim.item()
                else:
                    loss = global_sim
                    metrics["local_similarity"] = float("nan")
            else:
                loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        momentum = 0.9 * momentum + grad / torch.norm(grad, p=1)


        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    saliency_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """PGD attack with Saliency Suppression Reconstruction Loss"""
    saliency_version = getattr(cfg.model, 'saliency_loss_version', 'v1')
    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Saliency Attack progress")
    total_steps = cfg.optim.steps

    for epoch in pbar:
        with torch.no_grad():
            if saliency_version == "semantic_distance":
                saliency_loss.set_ground_truth(target_crop(image_tgt), src_image=image_org)
            else:
                saliency_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        if saliency_version == "semantic_distance":
            all_global, all_local = ensemble_extractor.intermediate_forward(adv_image)
            global_sim = saliency_loss(all_global, all_local, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                all_global_crop, all_local_crop = ensemble_extractor.intermediate_forward(local_cropped)
                local_sim = saliency_loss(all_global_crop, all_local_crop, total_steps=total_steps)
                loss = local_sim
                metrics["local_similarity"] = local_sim.item()
            else:
                loss = global_sim
        else:
            adv_features, adv_features_local, adv_features_raw = ensemble_extractor(adv_image)
            global_sim = saliency_loss(adv_features, adv_features_raw, total_steps=total_steps)
            metrics["global_similarity"] = global_sim.item()

            if cfg.model.use_source_crop:
                local_cropped = source_crop(adv_image)
                local_features, local_features_local, local_features_raw = ensemble_extractor(local_cropped)
                if local_features_raw and sum(v.abs().sum() for v in local_features_raw.values()) > 0:
                    local_sim = saliency_loss(local_features, local_features_raw, total_steps=total_steps)
                    loss = local_sim
                    metrics["local_similarity"] = local_sim.item()
                else:
                    loss = global_sim
                    metrics["local_similarity"] = float("nan")
            else:
                loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # PGD: 同时考虑梯度符号和 step size
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )
        # 投影到 epsilon 球内
        delta.data = torch.clamp(delta.data, -cfg.optim.epsilon, cfg.optim.epsilon)

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


if __name__ == "__main__":
    main()