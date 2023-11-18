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
from ctransformers import AutoModelForCausalLM

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

def load_diffuser():
    # Load diffusers for inference
    vae = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
    vae.to("cuda", dtype=torch.float16)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        vae=vae,
        torch_dtype=torch.float16
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(1024)
    return pipeline, generator

def load_llm():
    model_id = "TheBloke/Llama-2-13B-GGML"
    model_file = 'llama-2-13b.ggmlv3.q4_1.bin'
    config = {'max_new_tokens': 77, 'repetition_penalty': 1.2, 'temperature': 0.9, 'stream': False, 'context_length':1024, 'top_k':150, 'top_p':0.95}
    llm = AutoModelForCausalLM.from_pretrained(model_id,
                                            model_file=model_file,
                                            model_type="llama",
                                            #lib='avx2', for cpu use
                                            gpu_layers=130, #110 for 7b, 130 for 13b
                                            **config
                                            )
    return llm

def load_data():
    test_data = pd.read_csv("./data/test/info.csv")
    train_data = pd.read_csv("./data/train/info.csv")
    test_data_trans = pd.read_csv("./data/test/info_trans.csv")
    train_data_trans = pd.read_csv("./data/train/info_trans.csv")
    caption_data = pd.read_csv("./data/train/caption.csv")
    caption_dict = dict()
    for i in range(len(caption_data)):
        caption_dict[caption_data.iloc[i]["image"]] = caption_data.iloc[i]["prompt"]

    return test_data, train_data, test_data_trans, train_data_trans, caption_dict

def main():
    args = parse_args()
    sim_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    test_data, train_data, test_data_trans, train_data_trans, caption_dict = load_data()

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
        similarity, k=2, dim=1, largest=True, sorted=True
    )  # q * topk

    pipeline, generator = load_diffuser()
    llm = load_llm()

    # Create save dir
    os.makedirs(args.output_dir, exist_ok=True)

    template = '''Describe the advertising image from the following advertisement\n\nAdvertisement: {}\nDescription of advertising photo: {}\n\nAdvertisement: {}\nDescription of advertising photo: {}\n\nAdvertisement: {}\nDescription of advertising photo: ''' 

    bs = 4
    data_len = len(test_data_trans)
    for i in tqdm(range(0, data_len, bs)):
        prompts = test_data_trans.iloc[i:min(i+bs, data_len)]["caption"].tolist()
        # descriptions = test_data_trans.iloc[i:min(i+bs, data_len)]["description"].tolist()
        # moreInfos = test_data_trans.iloc[i:min(i+bs, data_len)]["moreInfo"].tolist()
            
        init_image_paths = []
        save_paths = []
        for j in range(i, min(i+bs, data_len)):
            # Some value
            init_image_paths.append(train_data.iloc[int(indices[j][0])]["bannerImage"])
            save_paths.append(os.path.join(args.output_dir, test_data_trans.iloc[j]["bannerImage"]))

            # LLM prompts
            fewshot_in0 = train_data_trans.iloc[int(indices[j][0])]["caption"]
            fewshot_out0 = caption_dict[train_data_trans.iloc[int(indices[j][0])]["bannerImage"]]
            fewshot_in1 = train_data_trans.iloc[int(indices[j][1])]["caption"]
            fewshot_out1 = caption_dict[train_data_trans.iloc[int(indices[j][1])]["bannerImage"]]
            prompt = template.format(fewshot_in0, fewshot_out0, fewshot_in1, fewshot_out1, prompts[j-i])
            prompt = llm(prompt, stream=False)
            prompt = prompt.strip().split('\n')[0]
            prompts[j-i] = prompt

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