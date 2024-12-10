import os
import numpy as np
from datasets.base_dataset import BaseDataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob

class TinyImageNetDataset(BaseDataset):
    """A dataset class for TinyImageNet dataset.
    
    It assumes that the directory '/path/to/data/' contains the following directories:
        - train: contains the training images
        - val: contains the validation images
        - test: contains the test images
        - wnids.txt: contains the labels of the dataset
        - words.txt: contains the meaning of the labels
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=200)
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.
        
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dataroot = os.path.join(opt.dataroot)
        self.phase = opt.phase
        self.train_path = f'{self.dataroot}/train'
        self.val_path = f'{self.dataroot}/val/images'
        self.test_path = f'{self.dataroot}/test/images'

        class_code_path = f'{self.dataroot}/wnids.txt'  # 200 class code (sorted alphabetically)
        self.class_code = []
        with open(class_code_path, 'r') as f:
            lines = f.readlines()
            lines.sort()
            for idx, line in enumerate(lines):
                self.class_code.append((idx, line.strip()))

        class_map_path = f'{self.dataroot}/words.txt'   # map class code to class name
        class_map = {}
        with open(class_map_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_map[parts[0]] = parts[1]
        self.labels  = [class_map[code] for idx, code in self.class_code]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224, padding = 10),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean = (-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std = (1 / np.array(self.std)).tolist()
            ),
        ])

        self.load_data()
    
    def load_data(self):
        """Load the data from the disk."""

        if self.phase == 'train':
            self.trainset = ImageFolder(self.train_path, transform=self.transform)
        elif self.phase == 'val':
            val_map_path = f'{self.dataroot}/val/val_annotations.txt'   # map val image to class code
            val_map = {}
            with open(val_map_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    val_map[parts[0]] = parts[1]
            val_map_tuples = list(val_map.items())
            self.val_map_idx = [(f'{self.val_path}/{img}', next(idx for idx, code in self.class_code if code == cls)) for img, cls in val_map_tuples]

        elif self.phase == 'test':
            self.img_paths = glob(f'{self.test_path}/*.JPEG')
            self.labels = [None for _ in self.img_paths]
        else:
            raise ValueError(f'Invalid phase: {self.phase}')
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.phase == 'train':
            return len(self.trainset)
        elif self.phase == 'val':
            return len(self.val_map_idx)
        elif self.phase == 'test':
            return len(self.img_paths)
        else:
            raise ValueError(f'Invalid phase: {self.phase}')
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        
        Parameters:
            index (int) -- a random integer for data indexing
        
        Returns a dictionary that contains A and B.
            A (tensor) -- an image in the input domain
            B (int) -- class of the image
        """
        if self.phase == 'train':
            img, idx = self.trainset[index]
            return {'X': img, 'label': idx, 'indices': [idx]}
        elif self.phase == 'val':
            img_path, idx = self.val_map_idx[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return {'X': img, 'label': idx, 'indices': [idx]}
        elif self.phase == 'test':
            img_path = self.img_paths[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return {'X': img, 'label': None}
        else:
            raise ValueError(f'Invalid phase: {self.phase}')