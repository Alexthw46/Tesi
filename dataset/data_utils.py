import logging
import pickle

import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR100, FashionMNIST, Food101, EuroSAT, ImageNet
from torchvision.datasets import ImageFolder

import os

import numpy as np

log = logging.getLogger("app")

osj = os.path.join


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['Imagenet32_val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def get_dataset(data_dir, dataset, train, transform):
    Imagenet_Folder_with_indices = dataset_with_indices(ImageFolder)
    ImageNetDS_with_indices = dataset_with_indices(ImageNetDS)
    c100_idx = dataset_with_indices(CIFAR100)
    fm_idx = dataset_with_indices(FashionMNIST)
    f101_idx = dataset_with_indices(Food101)
    esat_idx = dataset_with_indices(EuroSAT)
    inet_idx = dataset_with_indices(ImageNet)
    dataset_key = dataset.lower()

    direct_loaders = {
        "cifar100": lambda: c100_idx(
            root=osj(data_dir, "cifar100"),
            train=train,
            transform=transform,
            download=True,
        ),
        "imagenet-sketch": lambda: inet_idx(
            osj(data_dir, "imagenet", "imagenet-sketch", "sketch"),
            transform=transform,
            split="val",
        ),
        "food-101": lambda: f101_idx(
            data_dir,
            transform=transform,
            split="test",
        ),
        "eurosat": lambda: esat_idx(
            data_dir,
            transform=transform,
            download=True,
        ),
        "fashion-mnist": lambda: fm_idx(
            root=data_dir,
            train=False,
            transform=transform,
            download=True,
        ),
    }

    folder_datasets = {
        "fruits360": osj("fruits-360", "Test"),
        "lsun-scene": osj("lsun", "scene"),
        "fashion1m": osj("fashion1M", "clean_data"),
        "imagenet": osj("imagenet", "imagenetv1", "val"),
        "objectnet": osj("objectnet-1.0", "images"),
        "imagenet-c1": osj("imagenet", "imagenet-c", "fog", "1"),
        "imagenet-c2": osj("imagenet", "imagenet-c", "contrast", "2"),
        "imagenet-c3": osj("imagenet", "imagenet-c", "snow", "3"),
        "imagenet-c4": osj("imagenet", "imagenet-c", "gaussian_blur", "4"),
        "imagenet-c5": osj("imagenet", "imagenet-c", "saturate", "5"),
        "imagenetv2": osj("imagenet", "imagenetv2", "imagenetv2-matched-frequency-format-val"),
        "office31-amazon": osj("office31", "amazon", "images"),
        "office31-dslr": osj("office31", "dslr", "images"),
        "office31-webcam": osj("office31", "webcam", "images"),
        "officehome-product": osj("officehome", "Product"),
        "officehome-realworld": osj("officehome", "RealWorld"),
        "officehome-art": osj("officehome", "Art"),
        "officehome-clipart": osj("officehome", "Clipart"),
    }

    if dataset_key in direct_loaders:
        return direct_loaders[dataset_key]()

    if dataset_key in folder_datasets:
        return Imagenet_Folder_with_indices(
            osj(data_dir, folder_datasets[dataset_key]),
            transform=transform,
        )

    raise NotImplementedError("Please add support for %s dataset" % dataset)


def split_idx(y_true, num_classes=1000):
    classes_idx = []

    y_true = np.array(y_true)

    for i in range(num_classes):
        classes_idx.append(np.where(y_true == i)[0])

    return classes_idx
