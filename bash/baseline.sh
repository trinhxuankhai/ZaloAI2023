config='configs/baseline.yaml'

accelerate launch --mixed_precision="fp16" train.py \
                  --config $config
                  --output_dir checkpoints/baseline \