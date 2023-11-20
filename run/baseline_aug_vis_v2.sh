config='configs/baseline_aug_vis_v2.yaml'

CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_vis.py \
                                         --config $config \
                                         --output_dir "checkpoints/baseline_aug_vis_v2"
 