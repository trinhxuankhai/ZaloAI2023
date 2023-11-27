import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DiffusionPipeline

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
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
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
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6
    )

    args = parser.parse_args()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    args.cfg = cfg
    print(f"====> load config from {args.config}")
   
    return args, cfg    

def main():
    args, cfg = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAIN.GRADIENT_ACCUMULATION_STEP,
        mixed_precision=args.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.TRAIN.SEED is not None:
        set_seed(cfg.TRAIN.SEED)

    ####################################################################
    with open('./data/test/llava_prompt.json', 'r') as f:
        explicit_prompt = json.load(f)

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.MODEL.NAME, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(cfg.MODEL.NAME, subfolder="vae", revision=args.revision)   
    unet = UNet2DConditionModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="unet", revision=args.revision
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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

    text_lora_parameters = LoraLoaderMixin._modify_text_encoder(
        text_encoder, dtype=torch.float32, rank=cfg.MODEL.RANK
    )


    # Prepare everything with our `accelerator`.
    lora_layers, text_encoder = accelerator.prepare(
        lora_layers, text_encoder
    )

    # Potentially load in the weights and states from a previous save
    accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
    accelerator.load_state(args.resume_from_checkpoint)

    # Create save dir
    os.makedirs(args.output_dir, exist_ok=True)
   
    if accelerator.is_main_process:
        logger.info(
            f"Running validation..."
        )
        
        # create pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            cfg.MODEL.NAME,
            vae=vae.to(weight_dtype),
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(text_encoder),
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        # if args.prediction_type == "v_prediction":
        #     pipeline.scheduler = DDIMScheduler.from_config(
        #         pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        #     )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if cfg.TRAIN.SEED is not None:
            generator = generator.manual_seed(cfg.TRAIN.SEED)

        for save_path, prompt in tqdm(explicit_prompt.items()):
            save_path = os.path.join(args.output_dir, save_path)
            image = pipeline(prompt, generator=generator, num_inference_steps=30, height=536, width=1024).images[0]
            image = image.resize((1024, 533))
            image.save(save_path)

if __name__ == "__main__":
    main()
