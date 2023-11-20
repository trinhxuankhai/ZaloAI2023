config='configs/vis_1e4_30.yaml'

accelerate launch --gpu_ids 1 --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e4_30"