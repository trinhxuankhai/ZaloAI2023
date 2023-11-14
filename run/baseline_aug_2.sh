config='configs/baseline_aug.yaml'

accelerate launch --mixed_precision="fp16" train.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_aug"\
                  --resume_from_checkpoint "latest"