config='configs/sd_2_1_50.yaml'

accelerate launch --mixed_precision="fp16" tools/train.py \
                  --config $config \
                  --prediction_type "v_prediction" \
                  --output_dir "checkpoints/sd_2_1_50"