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


class PascalVocDataset(BaseDataset):
    """A dataset class for pascal_voc_2007 dataset.

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
        self.phase = opt.phase # get the phase: train, val, test
        self.dir = os.path.join(opt.dataroot)
        with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2007/ImageSets/Main/train.txt"), 'r') as file1:
            content1 = file1.read()
        self.train = content1.strip().split('\n') # get the list of train images
        with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2007/ImageSets/Main/val.txt"), 'r') as file2:
            content2 = file2.read()
        self.val = content2.strip().split('\n') # get the list of val images
        with open (os.path.join(self.dir, "test/VOCdevkit/VOC2007/ImageSets/Main/test.txt"), 'r') as file3:
            content3 = file3.read()
        self.test = content3.strip().split('\n') # get the list of test images
        self.labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # get the classes of the dataset

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
        # self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

    def load_data(self):
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/

        self.X = []
        self.labels = dict.fromkeys(self.labels, [])

        print(f"Loading \033[92m{self.phase}\033[0m data")
        
        if self.phase == 'train':
            dir = os.path.join(self.dir, "trainval/VOCdevkit/VOC2007") # directory to the train images
            for image in self.train:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the train images

            for i, class_ in enumerate(self.labels): # get the labels of each class for train
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_train.txt'), 'r') as file:
                    content = file.read()
                    s = content.strip().split()[1::2]
                    self.labels[class_] = list(map(lambda x: 1 if int(x) == 1 else 0.5 if int(x) == 0 else 0, s))

        elif self.phase == 'val':
            dir = os.path.join(self.dir, "trainval/VOCdevkit/VOC2007") # directory to the val images
            for image in self.val:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the val images

            for i, class_ in enumerate(self.labels): # get the labels of each class for val
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_val.txt'), 'r') as file:
                    content = file.read()
                    s = content.strip().split()[1::2]
                    self.labels[class_] = list(map(lambda x: 1 if int(x) == 1 else 0.5 if int(x) == 0 else 0, s))

        elif self.phase == 'test':
            dir = os.path.join(self.dir, "test/VOCdevkit/VOC2007") # directory to the test images
            for image in self.test:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the val images

            for i, class_ in enumerate(self.labels): # get the labels of each class for train
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_test.txt'), 'r') as file:
                    content = file.read()
                    s = content.strip().split()[1::2]
                    self.labels[class_] = list(map(lambda x: 1 if int(x) == 1 else 0, s))
        
        else:
            raise ValueError(f'Invalid phase: {self.phase}')

    def get_class_list(self, Y_class: torch.Tensor):
        assert isinstance(Y_class, torch.Tensor), "Y_class should be a tensor"
        class_list = torch.nonzero(Y_class).squeeze().tolist()
        if not isinstance(class_list, list):   
            class_list = [class_list]
        return Y_class

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
        Y_dict = {key: self.labels[key][index] for key in self.labels.keys()}
        Y_class = torch.tensor(list(Y_dict.values()),dtype=torch.float32)
        # return {'X': X_tensor, 'Y_class': Y_class, 'Y': Y_dict} # return the image and its class
        return {'X': X_tensor, 'Y_class': Y_class, 'Y': Y_dict} # return the image and the first class

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)