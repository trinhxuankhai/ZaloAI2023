config='configs/baseline_aug.yaml'

accelerate launch --mixed_precision="fp16" test.py \
                  --config $config \
                  --checkpoint_dir checkpoints/baseline_aug \
                  --output_dir inference/baseline_aug \
                  --resume_from_checkpoint "final-checkpoint"