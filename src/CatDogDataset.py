from torch.utils.data import Dataset
from PIL import Image
import torch

class CatDogDataset(Dataset):
    def __init__(self, data, target, transform):
        super().__init__()
        self.data = data
        self.len = len(self.data)
        self.transform = transform
        self.target = torch.from_numpy(target).long()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        image = self.transform(image)
        return (image, self.target[index])
