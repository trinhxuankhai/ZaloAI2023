config='configs/baseline_controlnet_aug.yaml'

accelerate launch --mixed_precision="fp16" test_controlnet.py \
                  --config $config \
                  --model_dir "checkpoints/baseline_controlnet_aug" \
                  --negative_prompt "blurry, pixelated, noisy, distorted, low-resolution images" \
                  --output_dir "inference/baseline_controlnet_aug_neg"  