config='configs/vis_1e5_40_cap.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train_vis_v2.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e5_40_cap"