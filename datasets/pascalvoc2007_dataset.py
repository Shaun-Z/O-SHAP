import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset, get_transform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import torchvision.transforms as transforms
import torch


class PascalVoc2007Dataset(BaseDataset):
    """A dataset class for pascal_voc_2012 dataset.

    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segmentation', action='store_true', help='if specified, load the segmentation dataset')
        parser.set_argument('--num_classes', type=int, default=20, help='number of classes in the dataset')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)  # call the default constructor of BaseDataset
        self.phase = opt.phase # get the phase: train, val, test
        self.dir = os.path.join(opt.dataroot)

        self.labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # get the classes of the dataset

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if opt.segmentation:    # Load the segmentation dataset
            with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"), 'r') as file_train:
                content1 = file_train.read()
            with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"), 'r') as file_val:
                content2 = file_val.read()
            # with open (os.path.join(self.dir, "test/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt"), 'r') as file_test:
            #     content3 = file_test.read()
        else:                   # Load the classification dataset
            with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2012/ImageSets/Main/train.txt"), 'r') as file_train:
                content1 = file_train.read()
            with open (os.path.join(self.dir, "trainval/VOCdevkit/VOC2012/ImageSets/Main/val.txt"), 'r') as file_val:
                content2 = file_val.read()
            # with open (os.path.join(self.dir, "test/VOCdevkit/VOC2012/ImageSets/Main/test.txt"), 'r') as file_test:
            #     content3 = file_test.read()

        self.train = content1.strip().split('\n') # get the list of train images
        self.val = content2.strip().split('\n') # get the list of val images
        # self.test = content3.strip().split('\n') # get the list of test images

        if self.phase == 'train':
            self.transform_mask = transforms.Compose([  # transform for the mask
                transforms.Resize(224),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224, padding = 10),
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([   # transform for the image
                self.transform_mask,
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform_mask = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),     # Please note that some mask values may be cropped out !!!
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([
                self.transform_mask,
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
        self.Y = []
        self.mask = []
        self.dicts = dict.fromkeys(self.labels, [])

        print(f"Loading \033[92m{self.phase}\033[0m data")
        
        if self.phase == 'train':
            dir = os.path.join(self.dir, "trainval/VOCdevkit/VOC2012") # directory to the train images
            if self.opt.segmentation:  # Load the segmentation dataset
                self.mask = [os.path.join(dir, "SegmentationClass", image + '.png') for image in self.train] 
            for image in self.train:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the train images
                self.Y.append(image)

            for i, class_ in enumerate(self.dicts): # get the labels of each class for train
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_train.txt'), 'r') as file:
                    content = file.read()
                    name = content.strip().split()[0::2]
                    s = content.strip().split()[1::2]
                    name_s_dict = dict(zip(name, s))
                    s_values = [name_s_dict[v] for v in self.train]
                    self.dicts[class_] = list(map(lambda x: 1 if int(x) == 1 else 0.5 if int(x) == 0 else 0, s_values))

        elif self.phase == 'val':
            dir = os.path.join(self.dir, "trainval/VOCdevkit/VOC2012") # directory to the val images
            if self.opt.segmentation:  # Load the segmentation dataset
                self.mask = [os.path.join(dir, "SegmentationClass", image + '.png') for image in self.val]
            for image in self.val:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the val images
                self.Y.append(image)

            for i, class_ in enumerate(self.dicts): # get the labels of each class for val
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_val.txt'), 'r') as file:
                    content = file.read()
                    name = content.strip().split()[0::2]
                    s = content.strip().split()[1::2]
                    name_s_dict = dict(zip(name, s))
                    s_values = [name_s_dict[v] for v in self.val]
                    self.dicts[class_] = list(map(lambda x: 1 if int(x) == 1 else 0.5 if int(x) == 0 else 0, s_values))

        elif self.phase == 'test':
            dir = os.path.join(self.dir, "test/VOCdevkit/VOC2012") # directory to the test images
            if self.opt.segmentation:  # Load the segmentation dataset
                self.mask = [os.path.join(dir, "SegmentationClass", image + '.png') for image in self.test]
            for image in self.test:
                self.X.append(os.path.join(dir, "JPEGImages", image + '.jpg')) # get the list of path to the val images
                self.Y.append(image)

            for i, class_ in enumerate(self.dicts): # get the labels of each class for train
                with open(os.path.join(dir, "ImageSets/Main", class_ + '_test.txt'), 'r') as file:
                    content = file.read()
                    name = content.strip().split()[0::2]
                    s = content.strip().split()[1::2]
                    name_s_dict = dict(zip(name, s))
                    s_values = [name_s_dict[v] for v in self.test]
                    self.dicts[class_] = list(map(lambda x: 1 if int(x) == 1 else 0, s_values))
        
        else:
            raise ValueError(f'Invalid phase: {self.phase}')

    '''
    Reminder:
    Don't call this function from dataloader.dataset.get_class_list(Y_class), because dataloader adds a dimention to the 1st dimension of the tensor.
    '''
    def get_class_list(self, Y_class: torch.Tensor):
        assert isinstance(Y_class, torch.Tensor), "Y_class should be a tensor"
        class_list = torch.nonzero(Y_class).squeeze().tolist()
        if not isinstance(class_list, list):   
            class_list = [class_list]
        return class_list

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
        Y = self.Y[index]
        Y_dict = {key: self.dicts[key][index] for key in self.dicts.keys()}
        Y_class = torch.tensor(list(Y_dict.values()),dtype=torch.float32)

        if self.opt.segmentation:   # return the [image], its [class] and the [mask]
            mask = Image.open(self.mask[index])
            mask = self.transform_mask(mask)
            return {'X': X_tensor, 'Y_class': Y_class, 'Y': Y, 'mask': mask}
        else:   # return the [image] and its [class]
            return {'X': X_tensor, 'Y_class': Y_class, 'Y': Y} # return the image and the first class

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)