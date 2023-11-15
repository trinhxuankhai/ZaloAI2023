config='configs/baseline_aug_vis.yaml'

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_aug_vis"
#                  --resume_from_checkpoint "latest"