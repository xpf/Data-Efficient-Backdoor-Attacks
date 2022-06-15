import os
import numpy as np
import pickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class ImageNet10(data.Dataset):
    def __init__(self, root, train=True, trigger=None, transform=None):
        super(ImageNet10, self).__init__()
        self.root = root
        self.trigger = trigger
        self.transform = transform
        middle = 'train' if train else 'val'
        self.data, self.targets = [], []
        self.data = np.load(os.path.join(self.root, middle, 'data.npy'))
        with open(os.path.join(self.root, middle, 'target.txt'), 'rb') as fp:
            self.targets = pickle.load(fp)
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        backdoor, source = 0, target
        img = Image.fromarray(img)
        if self.trigger is not None: img, target, backdoor = self.trigger(img, target, backdoor, idx)
        if self.transform is not None: img = self.transform(img)
        img = self.toTensor(img)
        return img, target, backdoor, source, idx

    def __len__(self):
        return self.data.shape[0]
