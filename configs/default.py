from yacs.config import CfgNode as CN
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_C = CN()

dataset_path = os.path.join(BASE_DIR, "../data")

# DATA
_C.DATA = CN()
_C.DATA.DATA_DIR = dataset_path
_C.DATA.TRAIN_CSV_PATH = 'train/info_trans.csv'
_C.DATA.TEST_CSV_PATH = 'test/info_trans.csv'
_C.DATA.RESOLUTION = 512
_C.DATA.CENTER_CROP = False
_C.DATA.RANDOM_FLIP = False
_C.DATA.RANDOM_AUG = False
_C.DATA.COND_IMAGES = False

# Model specific configurations.
_C.MODEL = CN()
_C.MODEL.NAME = 'stabilityai/stable-diffusion-2-1'
_C.MODEL.XFORMERS = False # Whether or not to use xformers for memory efficient.
_C.MODEL.NOISE_OFFSET = 0 # https://www.crosslabs.org//blog/diffusion-with-offset-noise
_C.MODEL.RANK = 4 # Lora rank.
_C.MODEL.CONTROL_NET = False

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.SEED = 1337
_C.TRAIN.EPOCH = 100
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.MAX_NORM = 1
_C.TRAIN.GRADIENT_ACCUMULATION_STEP = 4

## Learning rate setting 
_C.TRAIN.LR = CN()
# Choose between ["linear", "cosine", "cosine_with_restarts", 
# "polynomial", "constant", "constant_with_warmup"]
_C.TRAIN.LR.MODE = "cosine" 
_C.TRAIN.LR.BASE_LR = 1e-4
_C.TRAIN.LR.WARMUP_EPOCH = 0
# Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
_C.TRAIN.LR.SCALE_LR = False 

## Optimizer setting 
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-2
_C.TRAIN.OPTIMIZER.EPSILON = 1e-08

# Validation configurations
_C.EVAL = CN()
_C.EVAL.EPOCH = 1

# Testing configurations
_C.TEST = CN()
_C.TEST.RESTORE_FROM = ""

def get_default_config():
    return _C.clone()