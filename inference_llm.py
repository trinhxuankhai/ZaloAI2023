import os
import random
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from diffusers import AutoPipelineForImage2Image, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

def load_data():
    test_data = pd.read_csv("./data/test/info.csv")
    train_data = pd.read_csv("./data/train/info.csv")
    test_data_trans = pd.read_csv("./data/test/info_trans.csv")
    train_data_trans = pd.read_csv("./data/train/info_trans.csv")

    return test_data, train_data, test_data_trans, train_data_trans

def main():
    args = parse_args()
    sim_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    test_data, train_data, test_data_trans = load_data()

    # Load captions
    test_captions = []
    for i in range(len(test_data)):
        test_captions.append(test_data.iloc[i]["caption"])

    train_captions = []
    for i in range(len(train_data)):
        train_captions.append(train_data.iloc[i]["caption"])

    # Extract embeddings
    test_embeds = []
    for test_caption in tqdm(test_captions):
        sample = torch.from_numpy(sim_model.encode([test_caption]))
        test_embeds.append(sample)

    train_embeds = []
    for train_caption in tqdm(train_captions):
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


    # Load diffusers for inference
    vae = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
    vae.to("cuda", dtype=torch.float16)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        vae=vae,
        torch_dtype=torch.float16
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(1024)
    pipeline.set_progress_bar_config(disable=True)

    # Create save dir
    os.makedirs(args.output_dir, exist_ok=True)

    bs = 4
    data_len = len(test_data_trans)
    prefix_prompt = "Create an advertising banner about "
    negative_prompt = "nude girl, deformed hands,  watermark, text, deformed fingers, blurred faces, irregular face, irrregular body shape, ugly eyes, deformed face, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, ugly eyes, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, 2 heads, 2 faces"
    for i in tqdm(range(0, data_len, bs)):
        prompts = test_data_trans.iloc[i:min(i+bs, data_len)]["caption"].tolist()
        descriptions = test_data_trans.iloc[i:min(i+bs, data_len)]["description"].tolist()
        moreInfos = test_data_trans.iloc[i:min(i+bs, data_len)]["moreInfo"].tolist()

        for k in range(len(prompts)):
            # prompts[k] = prefix_prompt + prompts[k] 
            # prompts[k] = prompts[k] + '. ' + descriptions[k] + '. ' + moreInfos[k]
            # prompts[k] = prefix_prompt + prompts[k] + '. ' + descriptions[k] + '. ' + moreInfos[k]
            prompts[k] = prefix_prompt + prompts[k] + ', description is ' + descriptions[k] + ' and more information is ' + moreInfos[k]
            
        init_image_paths = []
        save_paths = []
        for j in range(i, min(i+bs, data_len)):
            init_image_paths.append(train_data.iloc[int(indices[j])]["bannerImage"])
            save_paths.append(os.path.join(args.output_dir, test_data_trans.iloc[j]["bannerImage"]))

        init_images = []
        for init_image_path in init_image_paths:
            init_image = load_image(os.path.join('./data/train/images', init_image_path))
            init_images.append(init_image)

        # images = pipeline(prompts, image=init_images, generator=generator, num_inference_steps=30, strength=0.6, height=536, width=1024).images
        images = pipeline(prompts, image=init_images, 
                          generator=generator, num_inference_steps=30, 
                          strength=0.6, height=536, width=1024, negative_prompt=[negative_prompt]*len(prompts)).images

        for image, save_path in zip(images, save_paths):
            image = image.resize((1024, 533))
            image.save(save_path)

    # bs = 1
    # data_len = len(test_data_trans)
    # prefix_prompt = "Create an advertising banner about "
    # negative_prompt = "nude girl, deformed hands,  watermark, text, deformed fingers, blurred faces, irregular face, irrregular body shape, ugly eyes, deformed face, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, ugly eyes, squint, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, poorly framed, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, 2 heads, 2 faces"
    # for i in tqdm(range(0, data_len, bs)):
    #     prompts = test_data_trans.iloc[i:min(i+bs, data_len)]["caption"].tolist()
    #     descriptions = test_data_trans.iloc[i:min(i+bs, data_len)]["description"].tolist()
    #     moreInfos = test_data_trans.iloc[i:min(i+bs, data_len)]["moreInfo"].tolist()

    #     for k in range(len(prompts)):
    #         prompts[k] = prefix_prompt + prompts[k] + '. ' + descriptions[k] + '. ' + moreInfos[k]

    #     init_image_paths = []
    #     save_paths = []
    #     for j in range(i, min(i+bs, data_len)):
    #         init_image_paths.append(train_data.iloc[int(indices[j])]["bannerImage"])
    #         save_paths.append(os.path.join(args.output_dir, test_data_trans.iloc[j]["bannerImage"]))

    #     init_images = []
    #     for init_image_path in init_image_paths:
    #         init_image = load_image(os.path.join('./data/train/images', init_image_path))
    #         init_images.append(init_image)

    #     images = pipeline(prompts, image=init_images, 
    #                       generator=generator, num_inference_steps=30, 
    #                       strength=0.6, height=536, width=1024, 
    #                       negative_prompt=[negative_prompt], num_images_per_prompt=4).images

    #     rand = random.randint(0, 3)
    #     image = images[rand]
    #     image = image.resize((1024, 533))
    #     image.save(save_paths[0])

if __name__ == "__main__":
    main()