config='configs/sd_2_1_50.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train.py \
                  --config $config \
                  --checkpointing_steps 228 \
                  --checkpoints_total_limit 10 \
                  --output_dir "checkpoints/sd_2_1_50_cap/checkpoint-4332"