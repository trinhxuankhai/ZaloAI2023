config='configs/baseline_aug_vis_v3.yaml'

accelerate launch --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e4_20"