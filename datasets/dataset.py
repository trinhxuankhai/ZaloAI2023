import os
import torch
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
    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def test_collate_fn(samples):
    images = torch.stack([sample["images"] for sample in samples])
    captions = []
    for sample in samples:
        captions.append(sample["captions"])
    return {"images": images,
            "captions": captions}

class BannerDataset(Dataset):
    def __init__(self, data_cfg, tokenizer, transform=None, mode='train') -> None:
        super().__init__()
        assert (mode in ["train", "test"]), "Please specify correct data mode !"
        self.data_cfg = data_cfg
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_dir = data_cfg.DATA_DIR
        self.data_csv_path = data_cfg.TRAIN_CSV_PATH if mode == "train" else data_cfg.TEST_CSV_PATH
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
        
        # Load image
        image = default_loader(os.path.join(self.data_dir, self.mode, "images/", sample["bannerImage"]))
        if self.transform is not None:
            image = self.transform(image)

        # Load caption
        caption = sample["caption"]
        caption_ids = tokenize_caption(caption, self.tokenizer)
        print(caption_ids.shape)
        
        if self.mode == "train":
            return {"pixel_values": image, 
                    "input_ids": caption_ids}
        else:
            return {"images": image,
                    "captions": caption}