import os
import random
import torch
import numpy as np
import pandas as pd
from underthesea import pos_tag
from torch.utils.data import Dataset
from PIL import Image, ImageFont, ImageDraw
    
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
    
    # if samples[0]["conditioning_pixel_values"] is not None:
    #     conditioning_pixel_values = torch.stack([sample["conditioning_pixel_values"] for sample in samples])
    # else:
    #     conditioning_pixel_values = None
        
    return {"pixel_values": pixel_values, 
            "input_ids": input_ids, 
            # "conditioning_pixel_values": conditioning_pixel_values
            }

def test_collate_fn(samples):
    paths = []
    captions = []
    # conditioning_pixel_values = []
    for sample in samples:
        captions.append(sample["captions"])
        # conditioning_pixel_values.append(sample["conditioning_pixel_values"])
        paths.append(sample["paths"])

    return {"captions": captions,
            # "conditioning_pixel_values": conditioning_pixel_values,
            "paths": paths}

def create_cond_images(text, unicode_font):
    width=1024
    height=533
    back_ground_color=(0,0,0)
    font_color=(255,255,255)
    nouns = []
    for tag in pos_tag(text):
        if tag[1] == 'N':
            nouns.append(tag[0])
    
    margin_x = int(0.1*width)
    margin_y = int(0.1*height)
    cond_image = Image.new("RGB", (width,height), back_ground_color)
    draw = ImageDraw.Draw(cond_image)

    # Top left
    if random.random() > 0.5 and len(nouns) > 0:
        noun = random.choice(nouns)
        nouns.remove(noun)
        noun_x, noun_y = unicode_font.getsize(noun)
        pos_x, pos_y = random.randrange(margin_x, int(0.5*width-noun_x)), random.randrange(margin_y, int(0.5*height-noun_y))
        draw.text((pos_x, pos_y), noun, font=unicode_font, fill=font_color)

    # Top right
    if random.random() > 0.5 and len(nouns) > 0:
        noun = random.choice(nouns)
        nouns.remove(noun)
        noun_x, noun_y = unicode_font.getsize(noun)
        pos_x, pos_y = random.randrange(int(0.5*width), int(width-margin_x-noun_x)), random.randrange(margin_y, int(0.5*height-noun_y))
        draw.text((pos_x, pos_y), noun, font=unicode_font, fill=font_color)

    # Bottom left
    if random.random() > 0.5 and len(nouns) > 0:
        noun = random.choice(nouns)
        nouns.remove(noun)
        noun_x, noun_y = unicode_font.getsize(noun)
        pos_x, pos_y = random.randrange(margin_x, int(0.5*width-noun_x)), random.randrange(int(0.5*height), int(height-margin_y-noun_y))
        draw.text((pos_x, pos_y), noun, font=unicode_font, fill=font_color)

    # Bottom right
    if random.random() > 0.5 and len(nouns) > 0:
        noun = random.choice(nouns)
        nouns.remove(noun)
        noun_x, noun_y = unicode_font.getsize(noun)
        pos_x, pos_y = random.randrange(int(0.5*width), int(width-margin_x-noun_x)), random.randrange(int(0.5*height), int(height-margin_y-noun_y))
        draw.text((pos_x, pos_y), noun, font=unicode_font, fill=font_color)
    
    return cond_image

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

        # Load data
        self.data_csv_path = data_cfg.TRAIN_CSV_PATH if (mode == "train" or mode == "val") else data_cfg.TEST_CSV_PATH
        self.data = pd.read_csv(os.path.join(self.data_dir, self.data_csv_path))

        # # Load VN data
        # self.data_csv_path_vn = data_cfg.TRAIN_CSV_PATH_VN if (mode == "train" or mode == "val") else data_cfg.TEST_CSV_PATH_VN
        # self.data_vn = pd.read_csv(os.path.join(self.data_dir, self.data_csv_path_vn))

        # # Load font
        # self.unicode_font = ImageFont.truetype(data_cfg.FONT, data_cfg.FONT_SIZE)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        Data row format: 
        id | caption | description |   moreInfo  | bannerImage (path)
        3  | Áo .... | Mua ngay .. | Miễn phí ...| 3.jpg
        '''
        sample = self.data.iloc[index]

        # Load caption =0
        # caption = sample["caption"]
        caption = sample['caption'] + ', description is ' + sample['description'] + ' and more information is ' + sample['moreInfo']
        # caption_vn = self.data_vn.iloc[index]["caption"]
        
        if self.mode == "train" or self.mode == "val":
            # Load image
            image = default_loader(os.path.join(self.data_dir, "train", "images/", sample["bannerImage"]))
            if self.transform is not None:
                image = self.transform(image)
            caption_ids = tokenize_caption(caption, self.tokenizer)

            # # Load condition image
            # cond_image = None
            # if self.controlnet:
            #     cond_image = default_loader(os.path.join(self.data_dir, "train", "cond_images/", sample["bannerImage"]))
            #     if self.cond_transform is not None:
            #         cond_image = self.cond_transform(cond_image)

            if self.mode == "train":
                return {"pixel_values": image, 
                        "input_ids": caption_ids,
                        # "conditioning_pixel_values": cond_image
                        }
            else:            
                return {"captions": caption,
                        # "conditioning_pixel_values": cond_image,
                        "paths": sample["bannerImage"]}
        else:
            # cond_image = create_cond_images(caption_vn, self.unicode_font)
            return {"captions": caption,
                    # "conditioning_pixel_values": cond_image,
                    "paths": sample["bannerImage"]}
        

class BannerDatasetv2(Dataset):
    def __init__(self, data_cfg, tokenizer, transform=None, mode='train') -> None:
        super().__init__()
        assert (mode in ["train", "val", "test"]), "Please specify correct data mode !"
        self.mode = mode
        self.data_cfg = data_cfg
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_dir = data_cfg.DATA_DIR

        # Load data
        self.data_csv_path = data_cfg.TRAIN_CSV_PATH if (mode == "train" or mode == "val") else data_cfg.TEST_CSV_PATH
        org_data = pd.read_csv(os.path.join(self.data_dir, self.data_csv_path))

        data = []
        if mode == "train":
            for index in range(len(org_data)):
                sample = org_data.iloc[index]
                caption, description, moreInfo = sample['caption'], sample['description'], sample['moreInfo']
                image_path = sample['bannerImage']
                image_id = image_path[:-4]

                data.append(dict(caption=caption, 
                                 description=description,
                                 moreInfo=moreInfo,
                                 image_path=os.path.join(self.data_dir, "train", "images", image_path)))
                
                root = os.path.join(self.data_dir, "train", "craw_images", image_id)
                for path in os.listdir(root):
                    data.append(dict(caption=caption, 
                                 description=description,
                                 moreInfo=moreInfo,
                                 image_path=os.path.join(root, path)))
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        Data row format: 
        id | caption | description |   moreInfo  | bannerImage (path)
        3  | Áo .... | Mua ngay .. | Miễn phí ...| 3.jpg
        '''
        sample = self.data[index]

        # Load caption 
        caption = sample['caption'] + ', description is ' + sample['description'] + ' and more information is ' + sample['moreInfo']
        
        if self.mode == "train" or self.mode == "val":
            # Load image
            image = default_loader(sample['image_path'])
            if self.transform is not None:
                image = self.transform(image)
            caption_ids = tokenize_caption(caption, self.tokenizer)

            if self.mode == "train":
                return {"pixel_values": image, 
                        "input_ids": caption_ids}
            else:            
                return {"captions": caption,
                        "paths": sample["bannerImage"]}
        else:
            return {"captions": caption,
                    "paths": sample["bannerImage"]}