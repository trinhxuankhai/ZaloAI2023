config='configs/sd_2_1_30.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train.py \
                  --config $config \
                  --prediction_type="v_prediction" \
                  --output_dir "checkpoints/sd_2_1_30"