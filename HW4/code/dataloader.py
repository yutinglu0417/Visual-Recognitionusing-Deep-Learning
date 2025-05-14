import os
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
import random

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomAdjustSharpness(2, p=0.5),
    transforms.ColorJitter(brightness=(0.5, 1.5),
                           contrast=(0.5, 1.5),
                           saturation=(0.5, 1.5),
                           hue=(-18 / 255., 18 / 255.)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class HW4TrainDataset(Dataset):
    def __init__(self, root_dir):
        self.degraded_dir = os.path.join(root_dir, "degraded")
        self.clean_dir = os.path.join(root_dir, "clean")
        self.D = Degradation()
        # 取得所有 degraded 圖片路徑（rain/snow 不分）
        self.degraded_paths = \
            sorted(glob.glob(os.path.join(self.degraded_dir, "*.png")))
        self.patch_size = 128

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(self.patch_size),
        ])
        self.toTensor = ToTensor()

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = \
            img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = \
            img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        degraded_path = self.degraded_paths[idx]
        filename = os.path.basename(degraded_path)

        # 找出對應的 clean 圖片名稱
        if filename.startswith("rain-"):
            clean_filename = filename.replace("rain-", "rain_clean-")
        elif filename.startswith("snow-"):
            clean_filename = filename.replace("snow-", "snow_clean-")
        else:
            raise ValueError(f"Unknown degraded type in filename: {filename}")

        clean_path = os.path.join(self.clean_dir, clean_filename)

        degrad_img = \
            crop_img(
                np.array(Image.open(degraded_path).convert('RGB')),
                base=16
            )
        
        clean_img = \
            crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        degrad_patch, clean_patch = \
            random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return [clean_filename, 0], degrad_patch, clean_patch


class HW4TestDataset(Dataset):
    def __init__(self, root_dir):
        self.degraded_dir = os.path.join(root_dir, "degraded")
        self.degraded_paths = \
            sorted(glob.glob(os.path.join(self.degraded_dir, "*.png")))

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 轉成 (C,H,W)
        ])

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        path = self.degraded_paths[idx]
        filename = os.path.basename(path)
        # print(filename)
        img = self.transform(Image.open(path).convert("RGB"))
        return filename, img


def get_loader(root_dir='./train',
               batch_size=2,
               num_workers=4,
               train_ratio=0.8):

    # Train
    train_dataset = HW4TrainDataset("../hw4_realse_dataset/train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Test
    test_dataset = HW4TestDataset("../hw4_realse_dataset/test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader
