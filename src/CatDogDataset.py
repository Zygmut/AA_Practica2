from torch.utils.data import Dataset
from skimage.io import imread


class CatDogDataset(Dataset):
    def __init__(self, data, target, transform):
        super().__init__()
        self.data = data
        self.len = len(self.data)
        self.transform = transform
        self.target = target

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.transform(imread(self.data[index])), self.target[index])
