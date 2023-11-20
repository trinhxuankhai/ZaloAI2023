config='configs/baseline.yaml'

accelerate launch --mixed_precision="fp16" test.py \
                  --config $config \
                  --inference_steps 15 \
                  --checkpoint_dir checkpoints/baseline \
                  --output_dir inference/baseline \
                  --resume_from_checkpoint "final-checkpoint"