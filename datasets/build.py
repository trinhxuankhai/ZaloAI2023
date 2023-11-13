import torch
from torchvision import transforms
from .dataset import BannerDataset, train_collate_fn, test_collate_fn
from .random_aug import RandAugment

def build_dataloader(cfg, tokenizer):
    train_transform = transforms.Compose(
        [
            transforms.Resize(cfg.DATA.RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.DATA.RESOLUTION) if cfg.DATA.CENTER_CROP else transforms.RandomCrop(cfg.DATA.RESOLUTION),
            transforms.RandomHorizontalFlip() if cfg.DATA.RANDOM_FLIP else transforms.Lambda(lambda x: x),
            RandAugment(n=5, m=10) if cfg.DATA.RANDOM_AUG else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    cond_train_transform = transforms.Compose(
        [
            transforms.Resize(cfg.DATA.RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.DATA.RESOLUTION),
            transforms.ToTensor(),
        ]
    )

    train_dataset = BannerDataset(cfg.DATA, tokenizer, transform=train_transform, cond_transform=cond_train_transform, mode="train")
    test_dataset = BannerDataset(cfg.DATA, tokenizer, transform=None, cond_transform=None, mode="test")
    val_dataset = BannerDataset(cfg.DATA, tokenizer, transform=train_transform, cond_transform=None, mode="val")
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, 10)))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collate_fn,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    return train_dataloader, test_dataloader, val_dataloader

