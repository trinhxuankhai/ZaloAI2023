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
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from configs.default import get_default_config
#from .evaluation.metrics import ZaloMetric

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Trainning script")
    parser.add_argument('--config', default="configs/baseline.yaml", type=str, help='config_file')
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
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
    
    args = parser.parse_args()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    args.cfg = cfg
    print(f"====> load config from {args.config}")
   
    return args, cfg


def main():
    args, cfg = parse_args()
    accelerator = Accelerator()

    controlnet = ControlNetModel.from_pretrained(os.path.join(args.model_dir, f"final-model"))
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.MODEL.NAME, subfolder="tokenizer", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="unet", revision=args.revision
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
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

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # DataLoaders creation:
    _, test_dataloader, val_dataloader = build_dataloader(cfg, tokenizer)

    # Prepare everything with our `accelerator`.
    lora_layers, controlnet, val_dataloader = accelerator.prepare(
        lora_layers, controlnet, val_dataloader
    )    
    
    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=0,
        desc="Steps",
    )

    logger.info(
        f"Running testing......"
    )

    # create pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.MODEL.NAME,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if cfg.TRAIN.SEED is not None:
        generator = generator.manual_seed(cfg.TRAIN.SEED)
    
    # create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "cond_images"), exist_ok=True)
        
    start = time.time()
    for sample in test_dataloader:
        save_paths = []
        save_cond_paths = []
        for path in sample["paths"]:
            save_paths.append(os.path.join(args.output_dir, "images", path))
            save_cond_paths.append(os.path.join(args.output_dir, "cond_images", path))
            
        with torch.autocast("cuda"):
            images = pipeline(sample["captions"], sample["conditioning_pixel_values"], num_inference_steps=20, generator=generator, negative_prompt=[args.negative_prompt]*len(sample["captions"])).images

        for i, (image, save_path) in enumerate(zip(images, save_paths)):
            image = image.resize((1024, 533))
            image.save(save_path)
            sample["conditioning_pixel_values"][i].save(save_cond_paths[i])

        progress_bar.update(1)
    
    end = time.time()
    logger.info(
        f"Total inference time: {end-start} s"
    )
if __name__ == "__main__":
    main()
