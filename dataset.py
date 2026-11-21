"""只在官方 train 内划分；最终测试集通过独立接口延迟读取。"""
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T


class SelectChannel:
    def __init__(self, channel):
        self.channel = channel

    def __call__(self, tensor):
        return tensor[self.channel:self.channel + 1].repeat(3, 1, 1)


def build_transform(config, training=False):
    size = config.image_size
    ops = ([T.RandomResizedCrop(size, scale=config.crop_scale), T.RandomHorizontalFlip(config.flip_probability)]
           if training and config.augment else [T.Resize(size), T.CenterCrop(size)])
    if config.grayscale:
        ops += [T.Grayscale(num_output_channels=3)]
    ops += [T.ToTensor()]
    if config.channel is not None:
        ops += [SelectChannel(config.channel)]
    if config.normalize:
        ops += [T.Normalize(config.mean, config.std)]
    return T.Compose(ops)


def stratified_indices(targets, fraction=0.2, seed=42):
    rng = random.Random(seed)
    train, valid = [], []
    for label in sorted(set(targets)):
        indices = [i for i, target in enumerate(targets) if target == label]
        rng.shuffle(indices)
        n = round(len(indices) * fraction)
        if not 0 < n < len(indices):
            raise ValueError('每类必须同时包含训练与内部验证样本')
        valid.extend(indices[:n])
        train.extend(indices[n:])
    return sorted(train), sorted(valid)


def build_internal_datasets(config):
    config.validate()
    root = Path(config.data_root) / 'train'
    train_source = datasets.ImageFolder(root, transform=build_transform(config, True))
    valid_source = datasets.ImageFolder(root, transform=build_transform(config))
    if len(train_source) != 9025 or len(train_source.classes) != 10:
        raise ValueError('官方 train 的类别数或图片数异常')
    assert train_source.samples == valid_source.samples
    train, valid = stratified_indices(train_source.targets, config.val_fraction, config.split_seed)
    # 记录相对文件名及标签摘要，防止数据顺序变化后悄悄复用旧划分
    manifest = '\n'.join(f'{Path(p).relative_to(root).as_posix()}:{y}' for p, y in train_source.samples)
    info = {'train_indices': train, 'validation_indices': valid,
            'class_to_idx': train_source.class_to_idx, 'classes': train_source.classes,
            'manifest_sha256': hashlib.sha256(manifest.encode()).hexdigest(),
            'split_seed': config.split_seed, 'val_fraction': config.val_fraction}
    return Subset(train_source, train), Subset(valid_source, valid), info


def build_final_test_dataset(config, expected_mapping):
    """仅在所有设计及模型选择结束后调用，不能用于调参。"""
    ds = datasets.ImageFolder(Path(config.data_root) / 'val', transform=build_transform(config))
    if ds.class_to_idx != expected_mapping or len(ds) != 3929:
        raise ValueError('最终测试集类别映射或图片数异常')
    return ds


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, device, training=False):
    return DataLoader(dataset, batch_size=config.batch_for(device), shuffle=training,
                      num_workers=config.workers, pin_memory=device.type == 'cuda',
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config.seed))
