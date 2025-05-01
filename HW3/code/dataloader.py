import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import json


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


class InstanceSegDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = os.listdir(root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_path = os.path.join(self.root_dir, sample_name)
        # Load image
        img_path = os.path.join(sample_path, "image.tif")
        img = Image.open(img_path).convert("RGB")

        # Prepare masks and labels
        masks = []
        labels = []
        boxes = []

        for fname in os.listdir(sample_path):
            if fname.startswith("class") and fname.endswith(".tif"):
                match = re.match(r"class(\d+)_(\d+)\.tif", fname)
                if match:
                    label = int(match.group(1))
                    mask_path = os.path.join(sample_path, fname)
                    mask = Image.open(mask_path).convert("L")
                    mask_np = np.array(mask)
                    mask_np = (mask_np > 0).astype(np.uint8)
                    if mask_np.max() > 0:
                        mask_tensor = torch.tensor(mask_np, dtype=torch.uint8)
                        masks.append(mask_tensor)

                        # Calculate bounding box
                        pos = torch.nonzero(mask_tensor)
                        xmin = torch.min(pos[:, 1])
                        xmax = torch.max(pos[:, 1])
                        ymin = torch.min(pos[:, 0])
                        ymax = torch.max(pos[:, 0])
                        if xmax == xmin or ymax == ymin:
                            # print(xmin, ymin, xmax, ymax)
                            # num_ones = torch.sum(mask_tensor == 255).item()
                            continue
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(label)

        # Convert all to tensor
        if len(masks) == 0:
            # 若無 mask，回傳空的 target（可選擇丟棄此樣本）
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.stack(masks)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # To tensor
        img = TF.to_tensor(img)

        target = {
            # "image_id": sample_name,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target, sample_name


class TestInstanceSegDataset(Dataset):
    def __init__(self, image_dir='./test_release',
                 id_json_path='./test_image_name_to_ids.json',
                 transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        # 解析 JSON 成為 list of dicts
        with open(id_json_path, "r") as f:
            metadata_list = json.load(f)

        # 建立一個清單儲存所有圖像資訊
        self.samples = []
        for item in metadata_list:
            name = item["file_name"]
            image_id = item["id"]
            height = item["height"]
            width = item["width"]
            path = os.path.join(image_dir, name)
            if os.path.isfile(path):  # 確保圖像存在
                self.samples.append({
                    "name": name,
                    "id": image_id,
                    "path": path,
                    "height": height,
                    "width": width
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        image = TF.to_tensor(image)
        if self.transforms:
            image = self.transforms(image)

        return image, sample["id"], sample["name"]


def collate_fn(batch):
    return tuple(zip(*batch))


# 建立 DataLoader
def get_loader(root_dir='./train',
               batch_size=2,
               num_workers=4,
               train_ratio=0.8):
    dataset = InstanceSegDataset(root_dir)
    test_dataset = TestInstanceSegDataset(transforms=val_transform)

    generator = torch.Generator().manual_seed(42)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=generator)
    train_dataset.dataset.transforms = train_transform
    val_dataset.dataset.transforms = val_transform
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate_fn)
    valid_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_fn)

    return train_dl, valid_dl, test_dl


if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_loader()

    for image, target, name in train_dl:
        for index, t in enumerate(target):
            for b in t['boxes']:
                if (b[0] == b[1]):
                    print(b[0], b[1])
                    print(name[index])
