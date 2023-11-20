config='configs/vis_1e6_20.yaml'

accelerate launch --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e6_20"