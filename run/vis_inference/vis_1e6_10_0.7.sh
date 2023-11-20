config='configs/vis_1e6_10.yaml'

python3 test_viz.py --mixed_precision="fp16" --output_dir "inference/vis_finetune/vis_1e6_10_0.7" \
                    --config $config \
                    --resume_from_checkpoint "checkpoints/vis/vis_1e6_10/final-checkpoint" \
                    --strength 0.7