config='configs/baseline_aug_sag.yaml'

accelerate launch --mixed_precision="fp16" train.py \
                  --config $config \
                  --output_dir "checkpoints/baseline_aug_sag"\
                  --resume_from_checkpoint "latest"