config='configs/vis_1e5_10_crawl.yaml'

accelerate launch --gpu_ids 0 --mixed_precision="fp16" train_vis.py \
                  --config $config \
                  --output_dir "checkpoints/vis/vis_1e5_10_crawl"