import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from cfg import get_cfg

cfg = get_cfg()
    
## augmentation
def augmentation(image):
    p = np.random.uniform(0, 1)
    if cfg.dataset == "imagenet":
        crop_size = 224
    else:
        crop_size = 32
    
    gray = transforms.Grayscale()
    affine = transforms.RandomAffine(degrees=2)
    half_flip = transforms.RandomHorizontalFlip()
    flip = transforms.RandomHorizontalFlip(p=1)
    crop = transforms.RandomResizedCrop(size=crop_size)
    jitter = transforms.ColorJitter()
    rotation = transforms.RandomRotation(degrees=20)

    if p < (1/6):
        image = gray(image)
        image = half_flip(image)
    elif p < (2/6):
        image = affine(image)
    elif p < (3/6):
        image = flip(image)
    elif p < (4/6):
        image = crop(image)
    elif p < (5/6):
        image = jitter(image)
    else:
        image = rotation(image)

    return image

## Dataset
class CustomDataset(Dataset):
    def __init__(self, train, prob=cfg.aug_p, data_dir="./data"):
        if cfg.dataset == "cifar10": 
            self.data = datasets.CIFAR10(root=data_dir, train=train, download=True)
        elif cfg.dataset == "cifar100":
            self.data = datasets.CIFAR100(root=data_dir, train=train, download=True)
        elif cfg.dataset == "imagenet":
            if train:
                train = 'train'
            else:
                train = 'test'
            self.data = datasets.ImageNet(root=data_dir, train=train, download=True)
        self.prob = prob

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
        p = np.random.uniform(0, 1)
        if p <= self.prob:
            image = Image.fromarray(image)
            image = augmentation(image)
            image = np.asarray(image)
        else:
            image = image
        
        if cfg.img_size is not None:
            dim = (cfg.img_size, cfg.img_size)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def collate_fn(self, data):
        batch_x, batch_y = [], []

        for x, y in data:
            arr_x = np.array(x, dtype=np.uint8) # convert into numpy array
            timg_x = self.transform(arr_x)
            norm_x = timg_x / 255

            ## convert to tensor and transpose
            x_torch = torch.Tensor(np.array(norm_x, ndmin=4))
            x_torch = x_torch.permute(0,3,1,2)
            x_torch = np.squeeze(x_torch, axis=0)
            y_torch = torch.Tensor([y])

            batch_x.append(x_torch)
            batch_y.append(y_torch)
        
        batch_x = torch.stack(batch_x).float()
        batch_y = torch.cat(batch_y).long()

        return batch_x, batch_y