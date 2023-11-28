import os
import torch
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
    
def default_loader(path):
    return Image.open(path).convert('RGB')

def tokenize_caption(caption, tokenizer):
    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def train_collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat([sample["input_ids"] for sample in samples], dim=0)
        
    return {"pixel_values": pixel_values, 
            "input_ids": input_ids, 
            }

def test_collate_fn(samples):
    paths = []
    captions = []
    for sample in samples:
        captions.append(sample["captions"])
        paths.append(sample["paths"])

    return {"captions": captions,
            "paths": paths}

class BannerDataset(Dataset):
    def __init__(self, data_cfg, tokenizer, transform=None, mode='train') -> None:
        super().__init__()
        assert (mode in ["train", "val", "test"]), "Please specify correct data mode !"
        self.data_cfg = data_cfg
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_dir = data_cfg.DATA_DIR

        # Load data
        self.data_csv_path = data_cfg.TRAIN_CSV_PATH if (mode == "train" or mode == "val") else data_cfg.TEST_CSV_PATH
        self.data = pd.read_csv(os.path.join(self.data_dir, self.data_csv_path))

        with open(os.path.join(self.data_dir, "train", "llava_caption.json"), 'r') as f:
            self.train_caption = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        Data row format: 
        id | caption | description |   moreInfo  | bannerImage (path)
        3  | Áo .... | Mua ngay .. | Miễn phí ...| 3.jpg
        '''
        sample = self.data.iloc[index]

        # Load caption
        caption = sample['caption'] + ', description is ' + sample['description'] + ' and more information is ' + sample['moreInfo']
        
        if self.mode == "train" or self.mode == "val":
            caption = self.train_caption[sample["bannerImage"]]

            # Load image
            image = default_loader(os.path.join(self.data_dir, "train", "images/", sample["bannerImage"]))
            if self.transform is not None:
                image = self.transform(image)
            caption_ids = tokenize_caption(caption, self.tokenizer)

            if self.mode == "train":
                return {"pixel_values": image, 
                        "input_ids": caption_ids,
                        }
            else:            
                return {"captions": caption,
                        "paths": sample["bannerImage"]}
        else:
            return {"captions": caption,
                    "paths": sample["bannerImage"]}
        
