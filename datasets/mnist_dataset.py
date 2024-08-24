import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets

# class MnistDataset(BaseDataset):
#     """A dataset class for MNIST dataset.

#     It assumes that the directory '/path/to/data/' contains the following directories:
#     - train.csv
#     - test.csv
#     """

#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         return parser

#     def __init__(self, opt):
#         """Initialize this dataset class.

#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)  # call the default constructor of BaseDataset
#         self.dir = os.path.join(opt.dataroot)  # get the image directory
#         self.phase = opt.phase
#         self.load_data()

#     def load_data(self):
#         # random_seed = 2
#         """Load data into memory."""
#         data = pd.read_csv(os.path.join(self.dir, f'{self.phase}.csv'))
#         # load data from /path/to/data/
#         if self.phase == 'train':
#             self.Y_train = data['label'].values.reshape(-1)
#             X_train = data.drop(labels = ['label'], axis=1)
#             del data
#             X_train.isnull().any().describe()
#             X_train = X_train / 255.0
#             # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
#             self.X_train = X_train.values.reshape(-1,28,28,1)
#             # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
#             # Perform one-hot encoding on Y_train
#             # encoder = OneHotEncoder(categories='auto')
#             # Y_train = encoder.fit_transform(Y_train.values.reshape(-1, 1)).toarray()
#             # Split the train and the validation set for the fitting
#             # self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
#         else:
#             # test = pd.read_csv(os.path.join(self.dir, 'test.csv'))
#             test = data / 255.0
#             # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
#             test = test.values.reshape(-1,28,28,1)

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index -- a random integer for data indexing

#         Returns:
#             a dictionary of data with their names. It usually contains the data itself and its metadata information.
#         """
#         return {'X': self.X_train[index], 'Y': self.Y_train[index]}

#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.X_train)

class MnistDataset(BaseDataset):
    """A dataset class for MNIST dataset.

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
        self.dir = os.path.join(opt.dataroot)  # get the image directory
        self.phase = opt.phase
        self.train = True if self.phase == 'train' else False
        self.load_data()

    def load_data(self):
        """Load data into memory."""
        dataset = datasets.MNIST(root='./data', train=self.train, download=True, transform=None)
        self.X = dataset.data.reshape(-1, 28, 28, 1) / 255.0
        self.Y = dataset.targets

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        return {'X': self.X[index], 'Y': self.Y[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.X)
