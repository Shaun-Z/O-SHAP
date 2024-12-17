import os
import numpy as np
import torch
import json
from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import shap

class ImageNet50Dataset(BaseDataset):
    """A dataset class for ImageNet50 dataset.
    
    It assumes that the directory '/path/to/data/' contains the following directories:
        - train: contains the training images
        - val: contains the validation images
        - test: contains the test images
        - wnids.txt: contains the labels of the dataset
        - words.txt: contains the meaning of the labels
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=1000)
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.
        
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dataroot = None
        self.phase = 'test'

        # Getting ImageNet 1000 class names
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with open(shap.datasets.cache(url)) as file:
            self.labels = [v[1] for v in json.load(file).values()]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            # transforms.Lambda(nhwc_to_nchw),
            transforms.Lambda(lambda x: x * (1 / 255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            # transforms.Lambda(nchw_to_nhwc),
        ])

        self.inv_transform = transforms.Compose([
            # transforms.Lambda(nhwc_to_nchw),
            transforms.Normalize(
                mean = (-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std = (1 / np.array(self.std)).tolist()
            ),
            # transforms.Lambda(nchw_to_nhwc),
        ])

        self.load_data()

    def load_data(self):
        self.X, self.y = shap.datasets.imagenet50()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # img = self.transform(torch.Tensor(self.X[index]))
        img = self.transform(self.X[index])
        label = self.y[index]
        return {'X': img, 'label': label, 'indices': [label.astype(int)]}