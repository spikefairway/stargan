from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pandas as pd
import numpy as np
import pdb


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class OxfordCat(data.Dataset):
    """Dataset class for cat pictures from Oxford IIIT-PET dataset."""

    def __init__(self, image_dir, cond_tab_path, selected_attrs, transform, crop_size=256):
        """Initialize and preprocesss the OxfordCat dataset.

        Parameters
        ----------
        image_dir : string
            Path for image directory
        cond_tab_path : string
            Path for CSV file including condition information
        selected_attrs : list
            Attributes to learn or test
        transform : torchvision.transform
            Transformations
        crop_size : int
            Size to crop
        """            
        self.image_dir = image_dir
        self.cond_tab_path = cond_tab_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.crop_size = crop_size

        self.preprocess()

    def preprocess(self):
        """Preprocessing.
        """      
        # Extracted data with selected conditions
        tab0 = pd.read_csv(self.cond_tab_path)
        selected_attrs = \
            tab0['BreedName'].unique() if self.selected_attrs is None \
            else self.selected_attrs
        cond_tab = tab0.loc[tab0['BreedName'].isin(selected_attrs), :].copy()
        cond_tab.reset_index(inplace=True, drop=False)

        # Image path list
        self.img_path_list = [
            os.path.join(self.image_dir, '.'.join((img_name, 'jpg')))
            for img_name in cond_tab['ImageName'].values
        ]

        # Condition matrix
        self.cond_mat = cond_tab.loc[:, selected_attrs].values

        print('Finished preprocessing the Oxford Cat dataset')

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        c = self.cond_mat[[idx], :]
        transform = self.transform

        image = Image.open(img_path)

        img_size_min = np.minimum(*image.size)
        # Add Resize if image_size is smaller than crop_size
        if img_size_min < self.crop_size:
            transform = T.Compose([
                T.Resize(self.crop_size),
                transform
            ])

        return transform(image), torch.FloatTensor(c)

    def __len__(self):
        return len(self.img_path_list)

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

def get_loader_oxford(image_dir, cond_tab_path, selected_attrs, crop_size=256, image_size=128,
                      batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader for Oxford cat dataset.
    """
    # Transform
    transform = []
    if mode == 'train':
        transform.append(T.RandomCrop(crop_size))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.CenterCrop(crop_size))
    #pdb.set_trace()
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = OxfordCat(image_dir, cond_tab_path, selected_attrs, transform, crop_size=crop_size)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers
    )

    return data_loader