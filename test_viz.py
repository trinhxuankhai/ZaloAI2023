import os
import math
import wandb
import shutil
import logging
import argparse
import itertools
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from datasets.build import build_dataloader

import accelerate 
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers.utils import ContextManagers
from transformers import CLIPTextModel, CLIPTokenizer

from sentence_transformers import SentenceTransformer

import diffusers
from diffusers.utils import load_image
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import AutoencoderKL, DDPMScheduler, AutoPipelineForImage2Image, UNet2DConditionModel

from configs.default import get_default_config

logger = get_logger(__name__, log_level="INFO")

def preprocess_val():
    sim_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    test_data = pd.read_csv("./data/test/info.csv")
    train_data = pd.read_csv("./data/train/info.csv")
    test_data_trans = pd.read_csv("./data/test/info_trans.csv")

    # Load captions
    test_captions = []
    for i in range(len(test_data)):
        test_captions.append(test_data.iloc[i]["caption"])

    train_captions = []
    for i in range(len(train_data)):
        train_captions.append(train_data.iloc[i]["caption"])

    # Extract embeddings
    test_embeds = []
    for test_caption in test_captions:
        sample = torch.from_numpy(sim_model.encode([test_caption]))
        test_embeds.append(sample)

    train_embeds = []
    for train_caption in train_captions:
        sample = torch.from_numpy(sim_model.encode([train_caption]))
        train_embeds.append(sample)

    test_embeds = torch.cat(test_embeds, dim=0)
    train_embeds = torch.cat(train_embeds, dim=0)

    test_embeds = F.normalize(test_embeds, dim=1, p=2)
    train_embeds = F.normalize(train_embeds, dim=1, p=2)

    # Calculate similarity 
    similarity = torch.mm(test_embeds, train_embeds.t())
    _, indices = torch.topk(
        similarity, k=1, dim=1, largest=True, sorted=True
    )  # q * topk

    return test_data, train_data, test_data_trans, indices

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

    # Val
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        test_data, train_data, test_data_trans, indices = preprocess_val()

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.MODEL.NAME, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_single_file(cfg.MODEL.VAE)   
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
    
    # DataLoaders creation:
    train_dataloader, test_dataloader, val_dataloader = build_dataloader(cfg, tokenizer)


    # Prepare everything with our `accelerator`.
    lora_layers, text_encoder, train_dataloader, test_dataloader, val_dataloader = accelerator.prepare(
        lora_layers, text_encoder, train_dataloader, test_dataloader, val_dataloader
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
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            cfg.MODEL.NAME,
            vae=vae.to(weight_dtype),
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(text_encoder),
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
        
        bs = 4
        data_len = len(test_data_trans)
        for i in tqdm(range(0, data_len, bs)):
            prompts = test_data_trans.iloc[i:min(i+bs, data_len)]["caption"].tolist()
            descriptions = test_data_trans.iloc[i:min(i+bs, data_len)]["description"].tolist()
            moreInfos = test_data_trans.iloc[i:min(i+bs, data_len)]["moreInfo"].tolist()

            for k in range(len(prompts)):
                prompts[k] = prompts[k] + ', description is ' + descriptions[k] + ' and more information is ' + moreInfos[k]
                
            init_image_paths = []
            save_paths = []
            for j in range(i, min(i+bs, data_len)):
                init_image_paths.append(train_data.iloc[int(indices[j])]["bannerImage"])
                save_paths.append(os.path.join(args.output_dir, test_data_trans.iloc[j]["bannerImage"]))

            init_images = []
            for init_image_path in init_image_paths:
                init_image = load_image(os.path.join('./data/train/images', init_image_path))
                init_images.append(init_image)

            images = pipeline(prompts, image=init_images, generator=generator, num_inference_steps=30, strength=0.6, height=536, width=1024).images
            
            for image, save_path in zip(images, save_paths):
                image = image.resize((1024, 533))
                image.save(save_path)

if __name__ == "__main__":
    main()
