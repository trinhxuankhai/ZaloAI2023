import os
import torch
import pandas as pd
import googletrans
from PIL import Image
from torch.utils.data import Dataset

class Translation():
    def __init__(self, from_lang='vi', to_lang='en'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate. 
        self.__to_lang = to_lang
        self.translator = googletrans.Translator()

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase

        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation

        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text, dest=self.__to_lang).text
    
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
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def test_collate_fn(samples):
    captions = []
    for sample in samples:
        captions.append(sample["captions"])
    return {"captions": captions}

class BannerDataset(Dataset):
    def __init__(self, data_cfg, tokenizer, transform=None, mode='train') -> None:
        super().__init__()
        assert (mode in ["train", "test"]), "Please specify correct data mode !"
        self.data_cfg = data_cfg
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.translater = Translation()
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

        # Load caption
        caption = self.translater(sample["caption"])
        
        if self.mode == "train":
            # Load image
            image = default_loader(os.path.join(self.data_dir, self.mode, "images/", sample["bannerImage"]))
            if self.transform is not None:
                image = self.transform(image)
        
            caption_ids = tokenize_caption(caption, self.tokenizer)
        
            return {"pixel_values": image, 
                    "input_ids": caption_ids}
        else:
            return {"captions": caption}