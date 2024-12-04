import os
import numpy as np
import random

from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

class EuroSATDataset(BaseDataset):
    """
    A dataset class for EuroSAT dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=10)
        return parser
    
    def __init__(self, opt):
        """
        Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.dataroot = os.path.join(opt.dataroot)
        self.dataset_path = os.path.join(self.dataroot, "2750")

        self.mean = [0.3442, 0.3801, 0.4076]
        self.std = [0.2025, 0.1368, 0.1155]

        if self.phase == 'train':
            self.transform = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((32, 32)),
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
    
    def load_data(self):
        """
        Load data from the dataset.
        """
        random.seed(0)
        dataset_full = ImageFolder(self.dataset_path, transform=self.transform)
        self.labels = dataset_full.classes

        if self.phase == 'train':
            self.dataset, _ = train_test_split(dataset_full, test_size=0.1, random_state=42)
        elif self.phase == 'val':
            _, self.dataset = train_test_split(dataset_full, test_size=0.1, random_state=42)
        elif self.phase == 'test':
            self.dataset = dataset_full
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'X': img, 'label': label, 'indices': [label]}
    
    def __len__(self):
        return len(self.dataset)