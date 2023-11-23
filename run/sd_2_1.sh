config='configs/sd_2_1_50.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train_vis_v2.py \
                  --config $config \
                  --output_dir "checkpoints/sd_2_1_50"