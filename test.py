import os
import time
import torch
import argparse
from tqdm.auto import tqdm
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets.build import build_dataloader
from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from configs.default import get_default_config

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Trainning script")
    parser.add_argument('--config', default="configs/baseline.yaml", type=str, help='config_file')
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/baseline",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    args = parser.parse_args()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    args.cfg = cfg
    print(f"====> load config from {args.config}")
   
    return args, cfg


def main():
    args, cfg = parse_args()
    accelerator = Accelerator()

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.MODEL.NAME, subfolder="tokenizer", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="unet", revision=args.revision
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=cfg.MODEL.RANK,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    
    # DataLoaders creation:
    _, test_dataloader, _ = build_dataloader(cfg, tokenizer)

    # Prepare everything with our `accelerator`.
    lora_layers, test_dataloader = accelerator.prepare(
        lora_layers, test_dataloader
    )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.checkpoint_dir, path))
    
    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=0,
        desc="Steps",
    )

    logger.info(
        f"Running testing......"
    )

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.MODEL.NAME,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if cfg.TRAIN.SEED is not None:
        generator = generator.manual_seed(cfg.TRAIN.SEED)
    
    # create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    for sample in test_dataloader:
        save_paths = []
        for path in sample["paths"]:
            save_paths.append(os.path.join(args.output_dir, path))

        images = pipeline(sample["captions"], num_inference_steps=30, generator=generator, height=536, width=1024).images

        for image, save_path in zip(images, save_paths):
            image = image.resize((1024, 533))
            image.save(save_path)
        progress_bar.update(1)
    end = time.time()
    logger.info(
        f"Total inference time: {end-start} s"
    )
if __name__ == "__main__":
    main()
