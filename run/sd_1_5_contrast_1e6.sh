config='configs/sd_1_5_10.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train.py \
                  --config $config \
                  --prediction_type="v_prediction" \
                  --vae \
                  --output_dir "checkpoints/sd_1_5_10"