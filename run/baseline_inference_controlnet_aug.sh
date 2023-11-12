config='configs/baseline_controlnet_aug.yaml'

accelerate launch --mixed_precision="fp16" test_controlnet.py \
                  --config $config \
                  --checkpoint_dir checkpoints/baseline_controlnet_aug \
                  --output_dir inference/baseline_controlnet_aug                