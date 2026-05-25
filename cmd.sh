uv run vllm serve /home/gpuadmin/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23
uv run generate_adversarial_samples_saliency.py model.saliency_loss_version=v1  
uv run generate_adversarial_samples_foa_attack.py --config-name ensemble_3models_gram            
uv run generate_adversarial_samples_foa_attack.py --config-name ensemble_3models 
uv run python llava_evaluate.py -m blackbox.model_name=llava
uv run blackbox_text_generation.py blackbox.model_name=Qwen3.5-9B       
CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=3 uv run vllm serve /home/gpuadmin/Qwen3.5-9B --served-model-name Qwen3.5-9B
