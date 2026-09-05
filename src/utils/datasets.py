import torch
import torchvision
import numpy as np
import os
import os.path as osp
from PIL import Image
import torchvision
import torchvision.transforms as TF


DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def vae_transforms(split, aug='randcrop', img_size=256):
    t = []
    if split == 'train':
        if aug == 'randcrop':
            t.append(TF.Resize(img_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True))
            t.append(TF.RandomCrop(img_size))
        elif aug == 'centercrop':
            t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        else:
            raise ValueError(f"Invalid augmentation: {aug}")
        t.append(TF.RandomHorizontalFlip(p=0.5))
    else:
        t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        
    t.append(TF.ToTensor())

    return TF.Compose(t)


def cached_transforms(aug='tencrop', img_size=256, crop_ranges=[1.05, 1.10]):
    t = []
    if 'centercrop' in aug:
        t.append(TF.Lambda(lambda x: center_crop_arr(x, img_size)))
        t.append(TF.Lambda(lambda x: torch.stack([TF.ToTensor()(x), TF.ToTensor()(TF.functional.hflip(x))])))
    elif 'tencrop' in aug:
        crop_sizes = [int(img_size * crop_range) for crop_range in crop_ranges]
        t.append(TF.Lambda(lambda x: [center_crop_arr(x, crop_size) for crop_size in crop_sizes]))
        t.append(TF.Lambda(lambda crops: [crop for crop_tuple in [TF.TenCrop(img_size)(crop) for crop in crops] for crop in crop_tuple]))
        t.append(TF.Lambda(lambda crops: torch.stack([TF.ToTensor()(crop) for crop in crops])))
    else:
        raise ValueError(f"Invalid augmentation: {aug}")

    return TF.Compose(t)


class LatentsNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, img_size=None):
        self.dir = osp.join(root, split)
        self.pt_files = list(range(10000))# [osp.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith('.pt')]
        self.pt_files.sort()  
        if split != "train":
            self.pt_files = self.pt_files[:32]
        self.transform = TF.Compose([
                TF.Resize((224, 224), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
                TF.ToTensor(),
            ])

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path).squeeze()
        # data = torch.randn(16,32,96)
        # create pseudo image to maintain consistent data pipeline, image is needed for training the AR model not the tokenizer
        img = torch.randn(1)
        # sha256 = "test"
        # img = Image.open(f"/mnt/localssd/flux_images/{sha256}.png").convert('RGB')
        # img = self.transform(img)
        sha256 = os.path.basename(pt_path).split('.')[0] # unique identifier for each latent, otherwise generate a random
        return data, img, sha256


class SlotsNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, img_size=None):
        self.dir = osp.join(root, split) if split in ['train', 'val', 'test'] else root
        self.pt_files = [osp.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith('.pt')]
        # self.pt_files = list(range(10000))
        self.pt_files.sort()  
        if split == "val" or split == "test":
            self.pt_files = self.pt_files[:32]
        self.transform = TF.Compose([
                TF.Resize((224, 224), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
                TF.ToTensor(),
                TF.Normalize(DATASET_MEAN, DATASET_STD),
            ])

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path).squeeze()
        # data = torch.randn(512,32)
        # sha256 = "temp"
        sha256 = os.path.basename(pt_path).split('.')[0]
        img = Image.open(f"/mnt/localssd/flux_images/{sha256}.png").convert('RGB')
        img = self.transform(img)
        # sha256 = "test"
        return data, img, sha256


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.dir = root
        self.files = [osp.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith('.png')]
        self.files.sort()  
        self.transform = TF.Compose([
                TF.Resize((224, 224), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
                TF.ToTensor(),
                TF.Normalize(DATASET_MEAN, DATASET_STD),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        sha256 = os.path.basename(file_path).split('.')[0]
        img = Image.open(file_path).convert('RGB')
        img = self.transform(img)
        return img, sha256
