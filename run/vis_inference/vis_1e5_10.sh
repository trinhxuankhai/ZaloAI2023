config='configs/vis_1e5_10.yaml'

python3 test_viz.py --output_dir "inference/vis_finetune/vis_1e5_10" \
                    --config $config \
                    --resume_from_checkpoint "checkpoints/vis/vis_1e5_10/final-checkpoint"