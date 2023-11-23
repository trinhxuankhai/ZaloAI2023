config='configs/vis_1e6_10.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train_vis_v2.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e6_10_v2"