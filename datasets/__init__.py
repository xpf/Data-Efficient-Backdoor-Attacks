from os.path import join
from datasets.cifar10 import CIFAR10
from datasets.imagenet10 import ImageNet10
import torchvision.transforms as transforms

DATASETS = {
    'cifar10': CIFAR10,
    'imagenet10': ImageNet10,
}


def build_transform(train, img_size, crop, flip):
    transform = []
    transform.append(transforms.Resize((img_size, img_size)))
    transform.append(transforms.Pad(crop))
    if train:
        transform.append(transforms.RandomCrop((img_size, img_size)))
        if flip: transform.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        transform.append(transforms.CenterCrop((img_size, img_size)))
    transform = transforms.Compose(transform)
    return transform


def build_data(data_name, data_path, train, trigger, transform):
    data = DATASETS[data_name](root=join(data_path, data_name), train=train, trigger=trigger, transform=transform)
    return data
