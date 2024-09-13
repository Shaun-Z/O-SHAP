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
from collections import defaultdict



class Cifar10Dataset(BaseDataset):
    """A dataset class for Cifar10 dataset.

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
        self.dir = os.path.join(opt.dataroot)  # get the image directory: data/cifar10
        print('dirdir',self.dir)
        self.phase = opt.phase # get the phase: train, val, test
        # with open (os.path.join(self.dir, 'wnids.txt'), 'r') as file1:
        #     content1 = file1.read()
        # self.labels = content1.strip().split('\n') # get the labels of the dataset
        # with open (os.path.join(self.dir, 'words.txt'), 'r') as file2:
        #     content2 = file2.read()
        # self.labels_meaning = dict()
        # for line in content2.strip().split('\n'):
        #     label, meaning = line.split('\t')
        #     self.labels_meaning[label] = meaning # get the meaning of the labels
        self.load_data()
        # self.unpickle()
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))



    def load_data(self):
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        self.X = []
        self.Y = []
        self.labels_meaning = []

        print(f"Loading \033[92m{self.phase}\033[0m data")
        
        if self.phase == 'train':
            # 5 batch he cheng 1 ge
            dir1 = os.path.join(self.dir, 'data_batch_1') # directory to the 5 train images
            dir2 = os.path.join(self.dir, 'data_batch_2')
            dir3 = os.path.join(self.dir, 'data_batch_3')
            dir4 = os.path.join(self.dir, 'data_batch_4')
            dir5 = os.path.join(self.dir, 'data_batch_5')
            print('dir1',dir1)
            dict1 = unpickle(dir1)
            dict2 = unpickle(dir2)
            dict3 = unpickle(dir3)
            dict4 = unpickle(dir4)
            dict5 = unpickle(dir5)
            merged_dict = defaultdict(list)
            for d in (dict1, dict2, dict3, dict4, dict5):
                for key, value in d.items():
                    merged_dict[key].extend(value)
            self.X = merged_dict[b'data']  # get array here
            self.Y = merged_dict[b'filenames']
            self.labels_meaning = merged_dict[b'labels']

        elif self.phase == 'test':
            dir = os.path.join(self.dir, 'test_batch') # directory to the train images
            dict = unpickle(dir)
            self.X = dict[b'data']
            # self.X = glob(os.path.join(dir, '*.JPEG')) # get the list of path to the test images
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
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224, padding = 10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
        element = self.X[index]
        img = element.reshape((1, 3, 32, 32)).transpose(0, 2, 3, 1)  # .astype(np.float)
        PIL_img = Image.fromarray(img[0], 'RGB')
        X_tensor = transform(PIL_img)
        # path = self.X[index]
        # im = Image.open(path).convert("RGB") # read the image
        # X_tensor = transform(im)

        if self.phase == 'test':
            return {'X': X_tensor}
        else:
            return {'X': X_tensor, 'Y_class': self.labels_meaning[index], 'Y': self.Y[index]} # X_tensor : {num_batches, 3, 64, 64} , Y_one_hot : a list of labels
            # return {'X': self.X[index], 'Y_class': self.Y[index]}#数据格式类型需调整

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)