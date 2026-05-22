import os
import base64
import json
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import LlavaNextProcessor
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from google import genai
import openai
from openai import OpenAI
import anthropic
import vllm

from utils import (
    get_api_key,
    hash_training_config,
    setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
)

# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]

# Local LLaVA model path
LLAVA_MODEL_PATH = "/home/gpuadmin/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23"


def setup_gemini(api_key: str):
    return genai.Client(api_key=api_key)


def setup_claude(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


def setup_gpt4o(api_key: str):
    return OpenAI(
        api_key=api_key,
    )


def setup_llava():
    """Load LLaVA model with vLLM Engine on GPU."""
    llm = vllm.LLM(
        model=LLAVA_MODEL_PATH,
        trust_remote_code=True,
        max_model_len=1024,
        dtype="half",
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    # Build chat template from processor
    processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_PATH)
    chat_template = processor.tokenizer.chat_template or processor.tokenizer.default_chat_template
    return llm, chat_template


def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name

        if model_name == "gemini":
            api_key = get_api_key(model_name)
            self.client = setup_gemini(api_key)
        elif model_name == "claude":
            api_key = get_api_key(model_name)
            self.client = setup_claude(api_key)
        elif model_name == "gpt4o":
            api_key = get_api_key(model_name)
            self.client = setup_gpt4o(api_key)
        elif model_name == "llava":
            self.llm, self.chat_template = setup_llava()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str) -> str:
        if self.model_name == "gemini":
            return self._generate_gemini(image_path)
        elif self.model_name == "claude":
            return self._generate_claude(image_path)
        elif self.model_name == "gpt4o":
            return self._generate_gpt4o(image_path)
        elif self.model_name == "llava":
            return self._generate_llava(image_path)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gemini(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_llava(self, image_path: str) -> str:
        from vllm import SamplingParams
        processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_PATH)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one concise sentence, no longer than 20 words."},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        image = Image.open(image_path).convert("RGB")
        sampling_params = SamplingParams(max_tokens=100, temperature=0)
        outputs = self.llm.generate(
            {"promptrompt": prompt, "multi_modal_data": {"image": image}},
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text.strip()


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # Initialize wandb using shared utility
    setup_wandb(cfg)
    print(cfg)
    # Get config hash and setup paths
    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    # Get output paths using shared utility
    paths = get_output_paths(cfg, config_hash)
    ensure_dir(paths["desc_output_dir"])

    try:
        # Initialize description generator
        generator = ImageDescriptionGenerator(model_name=cfg.blackbox.model_name)

        # Process original and adversarial images
        tgt_descriptions = []
        adv_descriptions = []

        # Walk through the output directory for adversarial images
        print("Processing images...")
        for root, _, files in os.walk(paths["output_dir"]):
            for file in tqdm(files):
                # Check if file has valid image extension
                if any(
                    file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                ):
                    try:
                        # Get paths
                        adv_path = os.path.join(root, file)
                        # Extract just the filename without extension
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        # Try each valid extension for target image
                        target_found = False
                        for ext in VALID_IMAGE_EXTENSIONS:
                            tgt_path = os.path.join(
                                cfg.data.tgt_data_path, "1", filename_base + ext
                            )
                            if os.path.exists(tgt_path):
                                target_found = True
                                break

                        if target_found:
                            # Generate descriptions
                            tgt_desc = generator.generate_description(tgt_path)
                            adv_desc = generator.generate_description(adv_path)

                            tgt_descriptions.append((file, tgt_desc))
                            adv_descriptions.append((file, adv_desc))

                            # Log to wandb
                            wandb.log(
                                {
                                    f"descriptions/{file}/target": tgt_desc,
                                    f"descriptions/{file}/adversarial": adv_desc,
                                }
                            )

                        else:
                            print(
                                f"Target image not found for {filename_base} with any valid extension, skip it."
                            )

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # Save descriptions
        save_descriptions(
            tgt_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"target_{cfg.blackbox.model_name}.txt"
            ),
        )
        save_descriptions(
            adv_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"adversarial_{cfg.blackbox.model_name}.txt"
            ),
        )

        print(f"Descriptions saved to {paths['desc_output_dir']}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
