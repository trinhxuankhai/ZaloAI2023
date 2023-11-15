config='configs/baseline_aug.yaml'

accelerate launch --mixed_precision="fp16" -multi_gpu --num_processes=2 train.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_aug_v2"\
                  --resume_from_checkpoint "latest"