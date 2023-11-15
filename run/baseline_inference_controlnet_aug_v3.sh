config='configs/baseline_controlnet_aug.yaml'

accelerate launch --mixed_precision="fp16" test_controlnet.py \
                  --config $config \
                  --model_dir checkpoints/baseline_controlnet_aug_v3 \
                  --output_dir inference/baseline_controlnet_aug_v3               