import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset, get_transform
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import random
import torch


class SeverstalDataset(BaseDataset):
    """A dataset class for Severstal dataset.

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
        self.dir = os.path.join(opt.dataroot)  # get the image directory: "data/severstal"
        print(self.dir)
        self.phase = opt.phase  # get the phase: train, val, test
        self.train_annotation = pd.read_csv(os.path.join(self.dir, 'train.csv'))
        self.sample_submission = pd.read_csv(os.path.join(self.dir, 'sample_submission.csv'))
        self.load_data()
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

    def load_data(self):
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/
        self.X = [] # image dir
        self.Y = [] # image name
        self.Y_class = [] # ClassID of the image
        self.mask = []
        self.X_positive = []
        self.Y_positive = []
        self.Y_positive_class = []
        self.mask_positive = []


        print(f"Loading \033[92m{self.phase}\033[0m data")

        dir = os.path.join(self.dir, 'train_images')  # directory to the train images
        image_path = glob(os.path.join(dir, '*.jpg'))  # get the list of path to the train images
                                                       # ['./data/severstal/train_images/58ee62fd7.jpg', './data/severstal/train_images/eeffa4c49.jpg',···]
        image_name = os.listdir(dir)  # ['58ee62fd7.jpg', 'eeffa4c49.jpg',···]
        df_train_csv = pd.read_csv(os.path.join(self.dir, 'train.csv'))

        for name in image_name:
            if name in df_train_csv['ImageId'].values:
                self.Y.append(name)
                self.Y_class.append(df_train_csv[df_train_csv['ImageId'] == name]['ClassId'].tolist())
                self.mask.append(df_train_csv[df_train_csv['ImageId'] == name]['EncodedPixels'].tolist())
            else:
                self.Y_positive.append(name)
                self.Y_positive_class.append([0])
                self.mask_positive.append(['0'])
        negative_X = [path for path in image_path if any(y in path for y in self.Y)]  # 6666个缺陷钢铁
        self.X += negative_X
        positive_X = [path for path in image_path if all(y not in path for y in self.Y)]  # 5903个无缺陷钢铁
        self.X += positive_X
        self.Y += self.Y_positive
        self.Y_class += self.Y_positive_class
        self.mask += self.mask_positive
        # print('Y_class',self.X[:5],self.Y[:5],self.Y_class[:5],len(self.Y_class))

        # 随机相同顺序打乱
        random.seed(0)
        combined = list(zip(self.X, self.Y, self.Y_class, self.mask))
        random.shuffle(combined)
        X, Y, Y_class, mask = zip(*combined)
        self.X = list(X)
        self.Y = list(Y)
        self.Y_class = list(Y_class)
        self.mask = list(mask)
        categories = [0, 1, 2, 3, 4]
        one_hot_encoded = np.zeros((len(self.Y_class), len(categories)))
        for i, lst in enumerate(self.Y_class):
            for value in lst:
                one_hot_encoded[i][value] = 1
        self.Y_class = one_hot_encoded
        # print('Y_class',self.X[126:129],self.Y[126:129],self.Y_class[126:129],self.mask[126:129],len(self.Y_class),len(self.Y))
        X_train, X_val, Y_train, Y_val, Y_class_train, Y_class_val, mask_train, mask_val = train_test_split(
            self.X, self.Y, self.Y_class, self.mask, test_size=0.2, random_state=42)
        if self.phase == 'train':
            self.X = X_train
            self.Y = Y_train
            self.Y_class = Y_class_train
            self.mask = mask_train
        elif self.phase == 'val':
            self.X = X_val
            self.Y = Y_val
            self.Y_class = Y_class_val
            self.mask = mask_val
        elif self.phase == 'test':
            dir = os.path.join(self.dir, 'test_images')  # directory to the train images
            self.X = glob(os.path.join(dir, '*.jpg'))  # get the list of path to the test images
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
        if self.phase == 'train':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224, padding=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        path = self.X[index]
        im = Image.open(path).convert("RGB")  # read the image
        X_tensor = transform(im)

        if self.phase == 'test':
            return {'X': X_tensor}
        else:
            return {'X': X_tensor, 'Y_class': self.Y_class[index], 'Y': self.Y[index], 'Y_mask': self.mask[index]}  # return the image and its class

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)