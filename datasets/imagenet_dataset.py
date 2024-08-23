import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset
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
    - train.csv
    - test.csv
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
        self.load_data()

    def load_data(self):
        # define the extract_train_number function to make sure train glob will sort the files in the correct order
        def extract_train_number(path):
            folder_name = path.split('/')[-1]
            return int(folder_name[1:])
        # define the extract_val_number function to make sure val glob will sort the files in the correct order
        def extract_val_number(path):
            filename = path.split('/')[-1]
            number = filename.split('_')[1].split('.')[0]
            return int(number)
        
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/
        if self.phase == 'train':
            dir = os.path.join(self.dir, 'train') # directory to the train images
            directories = sorted(glob(os.path.join(dir, '*')),key = extract_train_number) # directories of the train images
            self.X_train = []
            self.Y_train = []
            for directory in directories:
                images = sorted(glob(os.path.join(directory, 'images', '*.JPEG')), key = extract_val_number) # add images of each directory in order
                self.X_train += images
                self.Y_train += [directory.split('/')[-1]] * len(images)

        elif self.phase == 'val':
            dir = os.path.join(self.dir, 'val') # directory to the val images
            self.X_val = sorted(glob(os.path.join(dir, 'images', '*.JPEG')), key = extract_val_number) # get the list of path to the test images
            with open(os.path.join(dir, 'val_annotations.txt'), 'r') as file:
                content = file.readlines() # read the content of the val_annotations.txt file
            self.Y_val = [line.split()[1] for line in content]

        else:
            dir = os.path.join(self.dir, 'test/images') # directory to the test images
            self.X_test = sorted(glob(os.path.join(dir, '*.JPEG')), key = extract_val_number) # get the list of path to the test images


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        path = self.X_train[index]
        im = Image.open(path).convert("RGB")
        X_train_tensor = transform(im).permute(1,2,0)
        return {'X': X_train_tensor, 'Y': self.Y_train[index]} # X_train : {num_batches, 64, 64, 3} , Y_train : a list of labels

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X_train)