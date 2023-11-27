config='configs/sd_2_1_50.yaml'

python3 test.py --mixed_precision="fp16" --output_dir "inference/sd_2_1_cap" \
                    --config $config \
                    --resume_from_checkpoint "checkpoints/sd_2_1_50/final-checkpoint"