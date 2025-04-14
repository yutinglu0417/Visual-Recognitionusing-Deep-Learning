import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import os
import torch
from torch.utils.data import DataLoader
import json
import cv2
import os


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAdjustSharpness(2, p=0.5),
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


class Dataset(Dataset):
    def __init__(self, t="train", transforms=None):
        self.t = t
        json_path = "./data/" + t + ".json"
        with open(json_path) as f:
            self.data = json.load(f)
        self.img_dir = "./data/" + t
        self.transforms = transforms
        self.id_to_filename = [img['file_name'] for img in self.data['images']]
        self.annotations = self.data['annotations']

    def __len__(self):
        return len(self.id_to_filename)

    def __getitem__(self, idx):
        img_path = self.img_dir + '/' + self.id_to_filename[idx]
        img = cv2.imread(img_path)

        boxes = []
        labels = []
        area = []
        iscrowd = []
        id = self.id_to_filename[idx].split('.')[0]
        for i in self.annotations:
            if (i['image_id'] == int(id)):
                x_max = i['bbox'][0] + i['bbox'][2]
                y_max = i['bbox'][1] + i['bbox'][3]
                boxes.append([i['bbox'][0], i['bbox'][1], x_max, y_max])
                labels.append(i['category_id'])
                area.append(i['area'])
                iscrowd.append(i['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        labels = labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'image_id': torch.as_tensor(int(id), dtype=torch.int64),
            'boxes': boxes,
            'labels': labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


test_dir = "./data/test"


class TestDataset(Dataset):
    def __init__(self, t="test", transform=None):
        self.t = t
        self.dir = "./data/" + t
        self.imgs = []
        self.transform = transform

        self.imgs = sorted(os.listdir(self.dir),
                           key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.dir + "/" + self.imgs[idx]
        image = cv2.imread(img_path)
        id = int(self.imgs[idx].split('.')[0])

        if self.transform:
            image = self.transform(image)

        return image, id


def collate_fn(batch):
    return tuple(zip(*batch))


def build_loader(batch_size):
    # create datasets
    train_dataset = Dataset(t="train", transforms=train_transform)
    valid_dataset = Dataset(t="valid", transforms=val_transform)
    test_dataset = TestDataset(transform=val_transform)
    # build DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_dataset = Dataset(t="train", transforms=train_transform)
    # train_dl, val_dl, test_dl = build_loader(batch_size=2)
    # test_dataset = TestDataset(t="test", transform=val_transform)
    print(train_dataset[0])
