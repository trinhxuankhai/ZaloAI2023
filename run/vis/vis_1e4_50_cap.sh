config='configs/vis_1e4_50_cap.yaml'

accelerate launch --gpu_ids 1 --mixed_precision="fp16" train_vis_v2.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e4_50_cap"