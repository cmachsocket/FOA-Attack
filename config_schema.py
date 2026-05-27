from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore



@dataclass
class WandbConfig:
    """Wandb-specific configuration"""

    entity: str = "???"  # fill your wandb entity
    project: str = "local_adversarial_attack"


@dataclass
class BlackboxConfig:
    """Configuration for blackbox model evaluation"""

    model_name: str = "gpt4v"  # Used as output file prefix AND vLLM model identifier (e.g. gpt4v, claude, gemini, deepseek-chat, qwen2.5-7b)
    batch_size: int = 1
    timeout: int = 30


@dataclass
class DataConfig:
    """Data loading configuration"""

    batch_size: int = 1
    num_samples: int = 1000
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./Ours"


@dataclass
class OptimConfig:
    """Optimization parameters"""

    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300


@dataclass
class ModelConfig:
    """Model-specific parameters"""

    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: tuple = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"  # Can be "cpu", "cuda:0", "cuda:1", etc.
    backbone: list = (
        "L336",
        "B16",
        "B32",
        "Laion",
    )  # List of models to use: L336, B16, B32, Laion
    use_gram_loss: bool = False        # 新增：启用 Gram Matrix 风格损失
    gram_loss_weight: float = 1.0      # 新增：Gram loss 权重
    
    # Saliency Attack 参数
    saliency_loss_version: str = "v1"   # 显著性损失版本: v1, v2, v3, semantic_distance
    saliency_ratio: float = 0.3         # 显著性比例：抑制前 X% 高显著区域
    alpha_schedule: str = "cosine"     # alpha 调度策略: linear, cosine, step
    ema_beta: float = 0.9             # EMA 平滑系数（每步动态 alpha 的平滑程度）


@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    attack: str = "fgsm"  # can be [fgsm, mifgsm, pgd]


# register config for different setting
@dataclass
class Ensemble3ModelsConfig(MainConfig):
    """Configuration for ensemble_3models.py"""

    data: DataConfig = field(default_factory=lambda: DataConfig(batch_size=1))
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            use_source_crop=True,
            use_target_crop=True,
            backbone=["B16", "B32", "Laion"],
        )
    )


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="ensemble_3models", node=Ensemble3ModelsConfig)
