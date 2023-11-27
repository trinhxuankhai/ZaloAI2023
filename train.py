import os
import math
import wandb
import json
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
from diffusers import AutoencoderKL, DDPMScheduler, AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler, UNet2DConditionModel, DiffusionPipeline

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
        default="./checkpoints/baseline",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
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
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
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
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAIN.GRADIENT_ACCUMULATION_STEP,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Val
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        train_data_trans = pd.read_csv("./data/train/info_trans.csv")
        with open('./data/train/llava_caption.json', 'r') as f:
            train_caption = json.load(f)
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.MODEL.NAME, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.MODEL.NAME, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.MODEL.NAME, subfolder="text_encoder", revision=args.revision
    )
    # vae = AutoencoderKL.from_single_file(cfg.MODEL.VAE)   
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
    vae.to(accelerator.device, dtype=torch.float32)
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
    
    # Configuration of optimizer
    if cfg.TRAIN.LR.SCALE_LR:
        cfg.TRAIN.LR.BASE_LR = (
            cfg.TRAIN.LR.BASE_LR * cfg.TRAIN.GRADIENT_ACCUMULATION_STEP * cfg.TRAIN.BATCH_SIZE * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW
    params_to_optimize = (itertools.chain(lora_layers.parameters(), text_lora_parameters))
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=cfg.TRAIN.LR.BASE_LR,
        betas=cfg.TRAIN.OPTIMIZER.BETAS,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        eps=cfg.TRAIN.OPTIMIZER.EPSILON,
    )

    # DataLoaders creation:
    train_dataloader, test_dataloader, val_dataloader = build_dataloader(cfg, tokenizer)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.TRAIN.GRADIENT_ACCUMULATION_STEP)
    max_train_steps = cfg.TRAIN.EPOCH * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        cfg.TRAIN.LR.MODE,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.LR.WARMUP_EPOCH * num_update_steps_per_epoch * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, text_encoder, optimizer, train_dataloader, test_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, text_encoder, optimizer, train_dataloader, test_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.TRAIN.GRADIENT_ACCUMULATION_STEP)
    max_train_steps = cfg.TRAIN.EPOCH * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.TRAIN.EPOCH = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = cfg.TRAIN.BATCH_SIZE * accelerator.num_processes * cfg.TRAIN.GRADIENT_ACCUMULATION_STEP

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.TRAIN.EPOCH}")
    logger.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.TRAIN.GRADIENT_ACCUMULATION_STEP}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.TRAIN.EPOCH):
        unet.train()
        text_encoder.train()
        vae.to(torch.float32)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if cfg.MODEL.NOISE_OFFSET:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += cfg.MODEL.NOISE_OFFSET * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.TRAIN.BATCH_SIZE)).mean()
                train_loss += avg_loss.item() / cfg.TRAIN.GRADIENT_ACCUMULATION_STEP

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(lora_layers.parameters(), text_lora_parameters)
                    )
                    accelerator.clip_grad_norm_(params_to_clip, cfg.TRAIN.MAX_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

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
            pipeline = pipeline.to(accelerator.device)
            # pipeline.scheduler = EulerAncestralDiscreteScheduler(
            #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            # )
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device)
            if cfg.TRAIN.SEED is not None:
                generator = generator.manual_seed(cfg.TRAIN.SEED)
            
            total_images = []
            total_captions = []

            bs = 1
            data_len = len(train_data_trans)
            for i in tqdm([972, 1343, 1205, 54, 950, 1042]):
                prompts = []                    
                save_paths = []
                for j in range(i, min(i+bs, data_len)):
                    save_paths.append(os.path.join(args.output_dir, train_data_trans.iloc[j]["bannerImage"]))
                    prompts.append(train_caption[train_data_trans.iloc[j]["bannerImage"]])

                images = pipeline(prompts, generator=generator, num_inference_steps=30, height=536, width=1024).images
                
                for image, prompt in zip(images, prompts):
                    total_images.append(image)
                    total_captions.append(prompt)

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "validation": [
                                wandb.Image(image, caption=caption)
                                for i, (image, caption) in enumerate(zip(total_images, total_captions))
                            ]
                        }
                    )
            del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"final-checkpoint")
        accelerator.save_state(save_path)
        logger.info(f"Saved model to {save_path}")
        
    accelerator.end_training()


if __name__ == "__main__":
    main()
