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


class ImageNetDataset(BaseDataset):
    """A dataset class for ImageNet dataset.

    It assumes that the directory '/path/to/data/' contains the following directories:
        - train: contains the training images
        - val: contains the validation images
        - test: contains the test images
        - wnids.txt: contains the labels of the dataset
        - words.txt: contains the meaning of the labels
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)  # call the default constructor of BaseDataset
        self.dir = os.path.join(opt.dataroot)  # get the image directory: data/tiny-imagenet
        self.phase = opt.phase # get the phase: train, val, test
        with open (os.path.join(self.dir, 'wnids.txt'), 'r') as file1:
            content1 = file1.read()
        self.labels = content1.strip().split('\n') # get the labels of the dataset
        with open (os.path.join(self.dir, 'words.txt'), 'r') as file2:
            content2 = file2.read()
        self.labels_meaning = dict()
        for line in content2.strip().split('\n'):
            label, meaning = line.split('\t')
            self.labels_meaning[label] = meaning # get the meaning of the labels
        self.load_data()
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
            ])
        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean = (-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std = (1 / np.array(self.std)).tolist()
            ),
        ])

    def load_data(self):
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/
        self.X = []
        self.Y = []

        print(f"Loading \033[92m{self.phase}\033[0m data")
        
        if self.phase == 'train':
            dir = os.path.join(self.dir, self.phase) # directory to the train images
            directories = os.listdir(dir) # directories of the train images
            for directory in directories:
                images = glob(os.path.join(dir, directory, 'images', '*.JPEG')) # get the list of path to the train images
                self.X += images
                self.Y += [directory] * len(images) # current directory is the label of the images
                self.Y_class = torch.tensor([self.labels.index(item) for item in self.Y], dtype=torch.long) # get the class of each image

        elif self.phase == 'val':
            dir = os.path.join(self.dir, self.phase) # directory to the val images
            self.X = glob(os.path.join(dir, 'images', '*.JPEG')) # get the list of path to the test images
            with open(os.path.join(dir, 'val_annotations.txt'), 'r') as file:
                content = file.readlines() # read the content of the val_annotations.txt file
            for image in self.X:
                self.Y.append(content[int(image.split('_')[-1].split('.')[0])].split()[1]) # look up the label of each image in val_annotations.txt
            self.Y_class = torch.tensor([self.labels.index(item) for item in self.Y], dtype=torch.long) # get the class of each image

        elif self.phase == 'test':
            dir = os.path.join(self.dir, self.phase, 'images') # directory to the test images
            self.X = glob(os.path.join(dir, '*.JPEG')) # get the list of path to the test images
            '''Revisions are required here'''
        
        else:
            raise ValueError(f'Invalid phase: {self.phase}')


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        path = self.X[index]
        im = Image.open(path).convert("RGB") # read the image
        X_tensor = self.transform(im)

        if self.phase == 'test':
            return {'X': X_tensor}
        else:
            return {'X': X_tensor, 'Y_class': self.Y_class[index], 'Y': self.Y[index]} # return the image and its class


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)