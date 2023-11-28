from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.utils import disable_torch_init
from tqdm import tqdm
import pandas as pd
import json
import os

template = '''Given image is advertising about following information: "{}". What is happening in the image ?'''

def main():
    # Load model
    disable_torch_init()
    model_path = "liuhaotian/llava-v1.5-7b"

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    
    # Train data
    train_dir = '/home/server-96gb-ailab/Desktop/HungAn/Khai/ZaloAI2023/data/train'
    train_data = pd.read_csv(os.path.join(train_dir, 'info_trans.csv'))
    train_images_dir = os.path.join(train_dir, 'images')

    llava_caption = dict()
    for i in tqdm(range(len(train_data))):
        sample = train_data.iloc[i]
        prompt = template.format(sample['caption'] + sample['description'])
        image_file = os.path.join(train_images_dir, sample['bannerImage'])

        args = type('Args', (), {
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 77,
        })()

        output = eval_model(args, model_name, tokenizer, model, image_processor, context_len)
        
        llava_caption[sample['bannerImage']] = output

    with open(os.path.join(train_dir, 'llava_caption.json'), 'w') as f:
        json.dump(llava_caption, f)
        
if __name__ == "__main__":
    main()
