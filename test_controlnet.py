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
from .evaluation.metrics import ZaloMetric

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
    controlnet, val_dataloader = accelerator.prepare(
        controlnet, val_dataloader
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

    
    #metric = ZaloMetric()
    start = time.time()
    for sample in val_dataloader:
        save_paths = []
        for path in sample["paths"]:
            save_paths.append(os.path.join(args.output_dir, path))

        images = pipeline(sample["captions"], num_inference_steps=args.inference_steps, generator=generator, height=536, width=1024).images

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
