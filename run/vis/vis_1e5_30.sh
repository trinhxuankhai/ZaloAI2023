config='configs/vis_1e5_30.yaml'

accelerate launch --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e5_30"