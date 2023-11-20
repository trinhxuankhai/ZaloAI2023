config='configs/baseline_aug_vis_v3.yaml'

accelerate launch --gpu_ids 1 --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_aug_vis_v3"
 