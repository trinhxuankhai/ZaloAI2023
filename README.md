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
        |- ...
    |- test/
        |- info.csv
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
- Perform generation to reproduce the result, the output image will be at inference/output directory:
```
bash run/inference.sh
```

## Method 2: Finetunning Stable Diffusion model
### Solution
- Our team's solution is to use [LLAVA](https://github.com/haotian-liu/LLaVA) model to captionning the trainning dataset and perform finetunning Stable Diffusion model on the trainning data.

