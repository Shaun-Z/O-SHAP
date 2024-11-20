import os
import numpy as np
from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class BrainTumorMRIDataset(BaseDataset):
    """
    A dataset class for brain_tumor_mri dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=4)
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
        self.train_set_path = os.path.join(self.dataroot, "Training")
        self.val_set_path = os.path.join(self.dataroot, "Testing")

        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        self.mean = [0.1738,  0.1738, 0.1738]
        self.std = [0.1894,  0.1894, 0.1894]

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
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
        if self.phase == 'train':
            data_path = self.train_set_path
        elif self.phase == 'val':
            data_path = self.val_set_path
        elif self.phase == 'test':
            data_path = self.val_set_path
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
        
        self.dataset = ImageFolder(data_path, transform=self.transform)
        self.labels = self.dataset.classes

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image, label = self.dataset[index]
        return {'X': image, 'label': label, 'indices': [label]} # {image, label (directly used for loss calculation), class (indices of label)}
    
    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            the total number of images in the dataset.
        """
        return len(self.dataset)