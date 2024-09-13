import torch
from torch.utils.data import Dataset
from PIL import Image

class TrafficSignDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        numpy_image = self.images[idx]
        pil_image = Image.fromarray((numpy_image * 255).astype('uint8'))
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label
