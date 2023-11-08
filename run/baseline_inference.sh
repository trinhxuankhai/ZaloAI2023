config='configs/baseline.yaml'

accelerate launch --mixed_precision="fp16" test.py \
                  --config $config \
                  --output_dir checkpoints/baseline \
                  --resume_from_checkpoint "final-checkpoint"