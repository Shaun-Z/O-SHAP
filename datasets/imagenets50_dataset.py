import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class ImageNetS50Dataset(BaseDataset):
    """
    A dataset class for ImageNetS50 dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segmentation', action='store_true', help='if specified, load the segmentation dataset')
        parser.set_defaults(num_classes=50)
        return parser

    def __init__(self, opt):
        """
        Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.segmentation = opt.segmentation

        # Default dataset paths
        if self.segmentation:
            self.train_path = os.path.join(self.dataroot, "train_semi")
            self.train_semi_seg_path = os.path.join(self.dataroot, "train-semi-segmentation")
            self.val_seg_path = os.path.join(self.dataroot, "validation-segmentation")
        else:
            self.train_path = os.path.join(self.dataroot, "train")

        self.val_path = os.path.join(self.dataroot, "validation")
        
        # Determine the data path based on the phase
        if self.phase in ['train']:
            self.data_path = self.train_path
            self.data_seg_path = self.train_semi_seg_path if self.segmentation else None
        elif self.phase in ['validation', 'val', 'test']:
            self.data_path = self.val_path
            self.data_seg_path = self.val_seg_path if self.segmentation else None
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        # Pre-calculated mean and std for ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = self.get_transforms()

        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean=(-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std=(1 / np.array(self.std)).tolist()
            ),
        ])

        self.load_data()

    def get_transforms(self):
        """
        Get appropriate transforms based on the phase.
        """
        if self.phase in ['train']:
            self.transform_mask = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform_mask = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        return transforms.Compose([
            self.transform_mask,
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def load_data(self):
        """
        Load data from the dataset.
        """
        self.dataset = ImageFolder(self.data_path, transform=self.transform)
        self.labels = self.dataset.classes

        self.masks = ImageFolder(self.data_seg_path, transform=self.transform_mask) if self.segmentation else None

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image, label = self.dataset[index]
        if self.segmentation:
            mask, _ = self.masks[index]
            return {'X': image, 'label': label, 'indices': [label], 'mask': mask}
        else:
            return {'X': image, 'label': label,
                    'indices': [label]}  # {image, label (directly used for loss calculation), class (indices of label)}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            the total number of images in the dataset.
        """
        return len(self.dataset)
