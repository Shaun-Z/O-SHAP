import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset, get_transform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import torch

class Cub200Dataset(BaseDataset):
    """A dataset class for the CUB-200 dataset.

    It assumes that the directory '/path/to/data/' contains:
    - images (directory with all images)
    - train.csv (CSV file with training data annotations)
    - test.csv (CSV file with test data annotations)
    - class_labels.txt (file with class labels)
    - images.txt (file listing all image filenames)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def _init_(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset._init_(self, opt)
        self.dir = os.path.join(opt.dataroot)
        self.phase = opt.phase
        
        # Load class labels
        with open(os.path.join(self.dir, 'class_labels.txt'), 'r') as file:
            self.labels = file.read().strip().split('\n')
        
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        
        # Load image filenames
        with open(os.path.join(self.dir, 'images.txt'), 'r') as file:
            self.image_files = file.read().strip().split('\n')
        
        # Load annotations
        if self.phase == 'train':
            self.annotations = pd.read_csv(os.path.join(self.dir, 'train.csv'))
        elif self.phase == 'test':
            self.annotations = pd.read_csv(os.path.join(self.dir, 'test.csv'))
        else:
            raise ValueError(f'Invalid phase: {self.phase}')
        
        self.load_data()
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

    def load_data(self):
        """Load data into memory."""
        print(f"Loading \033[92m{self.phase}\033[0m data")
        
        self.X = []
        self.Y = []

        for _, row in self.annotations.iterrows():
            image_path = os.path.join(self.dir, 'images', row['image'])
            self.X.append(image_path)
            self.Y.append(self.label_map[row['label']])
        
        self.Y_class = torch.tensor(self.Y, dtype=torch.long)

    def _getitem_(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224, padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        path = self.X[index]
        im = Image.open(path).convert("RGB")
        X_tensor = transform(im)

        if self.phase == 'test':
            return {'X': X_tensor}
        else:
            return {'X': X_tensor, 'Y_class': self.Y_class[index]}

    def _len_(self):
        """Return the total number of images in the dataset."""
        return len(self.X)