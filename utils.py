import os
import sys
import time
import random
import numpy as np
from PIL import Image
import kornia.augmentation as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import *
from backdoors import *


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Dataset configurations (mean, std, size, num_classes)
_dataset_name = ['cifar10']

_mean = {
    'cifar10':  [0.4914, 0.4822, 0.4465],
}

_std = {
    'cifar10':  [0.2023, 0.1994, 0.2010],
}

_size = {
    'cifar10':  (32, 32),
}

_num = {
    'cifar10':  10,
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[dataset]))
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))

        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[dataset]))

    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


# Get dataset
def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=train & augment)
    
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)

    return dataset


# Get model
def get_model(dataset, network):
    num_classes = _num[dataset]

    if network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    else:
        raise NotImplementedError

    return model


# Get backdoor class
def get_backdoor(config, device):
    attack = config['attack']
    if attack == 'badnet':
        backdoor = BadNets(config, device)
    elif attack == 'dfst':
        backdoor = DFST(config, device)
    else:
        raise NotImplementedError

    return backdoor


# Construct a customized dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        return img, lbl

    def __len__(self):
        return len(self.images)


# Data augmentation
class ProbTransform(nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(nn.Module):
    def __init__(self, shape):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(A.RandomCrop(shape, padding=4), p=0.8)
        self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
