import os
import numpy as np
import random

from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

class ArrowDataset(BaseDataset):
    """
    A dataset class for Arrow dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--test_size', type=float, default=0.2, help='The size of the test set')
        parser.set_defaults(num_classes=2)
        return parser
    
    def __init__(self, opt):
        """
        Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.dataroot = opt.dataroot
        self.dataset_path = os.path.join(self.dataroot)

        self.test_size = opt.test_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
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

        if self.phase == 'train':
            self.dataset, _ = train_test_split(dataset_full, test_size=self.test_size, random_state=42)
        elif self.phase == 'val':
            _, self.dataset = train_test_split(dataset_full, test_size=self.test_size, random_state=42)
        elif self.phase == 'test':
            self.dataset = dataset_full
        else:
            raise NotImplementedError(f'Phase {self.phase} is not implemented')

        self.labels = dataset_full.classes

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'X': img, 'label': label, 'indices': [label]}

    def __len__(self):
        return len(self.dataset)
