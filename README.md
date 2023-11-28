# ZaloAI2023: Banner Generation 

## Installation and System Architecture
- All of the result can be reproduced on a single 24GB GPU (GPU A5000 in our cases).
- Installation :
```
pip install -r requirements.txt
```

## Data preparation
Organize dataset folder as follows:
```
|- data/
    |- train/
        |- images/
        |- info.csv
        |- info_trans.csv # Translated train data
        |- ...
    |- test/
        |- info.csv
        |- info_trans.csv # Translated test data
        |- ...
```

## Data preprocessing
- Translation data to English:
```
bash run/translation.sh
```

## Method 1: Inference from pretrained Stable Diffusion model
### Solution
- Our team's solution is to use images in the training set as condition images to support image generation for the stable diffusion model. To retrieve images from the training set, we uses a pretrained model to extract embedding and then calculates cosine similarity to find the closest image to support image generation on the test set. 
- From our experiment, [Realistic Vision](https://civitai.com/models/4201/realistic-vision-v51) version of Stable Diffusion 1.5 can produce best realistic result.

### Step to reproduce result
- Perform generation to reproduce the result, the output image will be at "inference/method_1" directory:
```
bash run/inference.sh
```

### Limitation
- For this solution, our team can not achieve higher score than 0.39541 from Zalo AI Benchmark.
- With the duplication score, this method failed with dulplication score up to 0.90793.	 

## Method 2: Finetunning Stable Diffusion model
### Solution
- Our team's solution is to use [LLAVA](https://github.com/haotian-liu/LLaVA) model to captioning the trainning dataset and perform finetunning Stable Diffusion model on the trainning data. 
- During inference, we use LLM to generate more correct caption from test data.
- From our experiment, [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) can easily finetuned to adapt the trainning data.

### Step to reproduce result
- LLAVA for image captioning: [llava](preprocessing/README.md)
- Finetunning Stable Diffusion model:
```
bash run/train_sd.py
```
- Pre generate LLM captioning for inference:
```
python3 prompt_engineer/caption.py
```
- Perform generation to reproduce the result, the output image will be at "inference/method_2" directory:
```
bash run/inference_sd.sh
```

### Limitation
- For this solution, our team can achieve considerable score 0.39673 from Zalo AI Benchmark. 
- However, for this solution our inference time up to 3.xx hours due to the large inference time of Large Language Model.