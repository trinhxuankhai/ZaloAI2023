config='configs/baseline_aug_vis.yaml'

accelerate launch --mixed_precision="fp16" test_vis.py \
                  --config $config \
                  --output_dir inference/baseline_aug_vis \
                  --resume_from_checkpoint "latest" \
                  --checkpoint_dir "checkpoints/baseline_aug_vis" \ 