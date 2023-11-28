config='configs/sd_2_1_50_768.yaml'

accelerate launch --multi_gpu --gpu_ids "[0,1]" --mixed_precision="fp16" train.py \
                  --config $config \
                  --prediction_type "v_prediction" \
                  --checkpointing_steps 228 \
                  --checkpoints_total_limit 10 \
                  --resume_from_checkpoint "latest" \
                  --output_dir "checkpoints/sd_2_1_50_768"