#!/usr/bin/env python3


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path


class MyDataLoader(DataLoader):
    def __init__(self, image_paths, batch_size, shuffle=False, num_workers=0, width=224, height=224):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = MyDataset(self.image_paths, self.transform)

        super().__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
