import os
import csv
from tqdm import tqdm
from PIL import Image
import pandas as pd
from clip_interrogator import Config, Interrogator

caption_model_name = 'blip-large'
clip_model_name = 'ViT-L-14/openai'

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def main():
    folder_path = "./data/train/images"
    prompt_mode = 'best'
    train_data = pd.read_csv("./data/train/info.csv")
    ci.config.quiet = True

    prompts = []
    start = int(0.25*len(train_data))
    end = int(0.5*len(train_data))
    for idx in tqdm(range(start, end)):
        image_path = train_data.iloc[idx]["bannerImage"]
        image = Image.open(os.path.join(folder_path, image_path)).convert('RGB')
        prompt = image_to_prompt(image, prompt_mode)
        prompts.append(dict(prompt=prompt, image=image_path))
    
    csv_path = os.path.join("./data/train", 'caption2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['image', 'prompt'])
        for prompt in prompts:
            w.writerow([prompt["image"], prompt["prompt"]])

if __name__ == "__main__":
    main()