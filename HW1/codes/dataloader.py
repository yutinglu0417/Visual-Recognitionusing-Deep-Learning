from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 600)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 600)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset(Dataset):
    def __init__(self, t="train", transform=None):
        self.t = t
        self.dir = "./data/" + t
        img_ls = os.listdir(self.dir)
        self.transform = transform
        self.path = []
        for labels in img_ls:
            for img in os.listdir(self.dir + '/'+ labels):
                self.path.append(self.dir + '/'+ labels + '/' +img)
                
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img_path = self.path[idx]
        image = cv2.imread(img_path)
        
        if self.transform:
            image = self.transform(image)

        path_split = img_path.split('/')
        label = int(path_split[3])
        return image, label
    
test_dir = "./data/test"

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.test_dir = test_dir
        self.image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        name = img_path.split("/")
        name = name[-1].split("\\")
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, name[-1]
    

def build_loader(batch_size):
    # create datasets   
    train_dataset = Dataset(t="train", transform=train_transform)
    valid_dataset = Dataset(t="val", transform=val_transform)
    test_dataset = TestDataset(transform=val_transform)
    # build DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    train_dl, val_dl, test_dl = build_loader(batch_size=32)
    train_dataset = Dataset(t="train", transform=train_transform)
    
    for images, labels in train_dl:
        print(labels)
        break




