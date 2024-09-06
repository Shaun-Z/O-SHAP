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
    '''A dataset class for CUB200 dataset.
    
    It assumes that the directory '/path/to/data/' contains the following directories:
    - images
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot)
        self.phase = opt.phase

    def load_data(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

