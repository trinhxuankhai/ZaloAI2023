config='configs/baseline_controlnet_aug.yaml'

accelerate launch --mixed_precision="fp16" train_controlnet.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_controlnet_aug"\
                  --resume_from_checkpoint "latest"