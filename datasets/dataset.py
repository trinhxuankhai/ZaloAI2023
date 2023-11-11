import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
    
def default_loader(path):
    return Image.open(path).convert('RGB')

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def tokenize_caption(caption, tokenizer):
    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def train_collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat([sample["input_ids"] for sample in samples], dim=0)
    
    if samples[0]["conditioning_pixel_values"] is not None:
        conditioning_pixel_values = torch.stack([sample["conditioning_pixel_values"] for sample in samples])
    else:
        conditioning_pixel_values = None
        
    return {"pixel_values": pixel_values, 
            "input_ids": input_ids, 
            "conditioning_pixel_values": conditioning_pixel_values}

def val_collate_fn(samples):
    captions = []
    conditioning_pixel_values = []
    for sample in samples:
        captions.append(sample["captions"])
        conditioning_pixel_values.append(sample["conditioning_pixel_values"])
        
    return {"captions": captions,
            "conditioning_pixel_values": conditioning_pixel_values}

def test_collate_fn(samples):
    captions = []
    paths = []
    for sample in samples:
        captions.append(sample["captions"])
        paths.append(sample["paths"])
    return {"captions": captions,
            "paths": paths}

class BannerDataset(Dataset):
    def __init__(self, data_cfg, tokenizer, transform=None, cond_transform=None, mode='train') -> None:
        super().__init__()
        assert (mode in ["train", "val", "test"]), "Please specify correct data mode !"
        self.data_cfg = data_cfg
        self.transform = transform
        self.cond_transform = cond_transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.controlnet = data_cfg.COND_IMAGES
        self.data_dir = data_cfg.DATA_DIR
        self.data_csv_path = data_cfg.TRAIN_CSV_PATH if (mode == "train" or mode == "val") else data_cfg.TEST_CSV_PATH
        self.data_csv_path = os.path.join(self.data_dir, self.data_csv_path)
        self.data = pd.read_csv(self.data_csv_path)

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
        caption = sample["caption"]
        
        # Load condition image
        cond_image = None
        if self.controlnet:
            cond_image = default_loader(os.path.join(self.data_dir, "train", "cond_images/", sample["bannerImage"]))
            cond_image = np.array(cond_image)
            cond_image = auto_canny(cond_image)
            cond_image = cond_image[:, :, None]
            cond_image = np.concatenate([cond_image, cond_image, cond_image], axis=2)
            if self.cond_transform is not None:
                cond_image = self.cond_transform(cond_image)
        
        if self.mode == "train":
            # Load image
            image = default_loader(os.path.join(self.data_dir, self.mode, "images/", sample["bannerImage"]))
            if self.transform is not None:
                image = self.transform(image)
            caption_ids = tokenize_caption(caption, self.tokenizer)
            return {"pixel_values": image, 
                    "input_ids": caption_ids,
                    "conditioning_pixel_values": cond_image}
        elif self.mode == "val":            
            return {"captions": caption,
                    "conditioning_pixel_values": cond_image}
        else:
            return {"captions": caption,
                    "paths": sample["bannerImage"]}