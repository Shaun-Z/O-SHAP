import os
import json
import numpy as np
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from PIL import Image

class MalariaDataset(BaseDataset):
    """
    A dataset class for malaria dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=2)  # Binary classification: red blood cell (1) vs others (0)
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

        # Paths to the JSON files containing annotations
        self.train_json = os.path.join(self.dataroot, "training.json")
        self.test_json = os.path.join(self.dataroot, "test.json")

        # Path to the images folder
        self.images_dir = os.path.join(self.dataroot, "images")

        # Pre-calculated mean and std (adjust based on dataset)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Define transforms for training and testing phases
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
                mean=(-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std=(1 / np.array(self.std)).tolist()
            ),
        ])

        self.load_data()

    def load_data(self):
        """
        Load data from the JSON files.
        """
        if self.phase == 'train':
            json_path = self.train_json
        elif self.phase in ['val', 'test']:
            json_path = self.test_json
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Extract file paths, labels, and bounding boxes from JSON
        self.image_paths = [
            os.path.join(self.images_dir, item['image']['pathname'].lstrip('/').replace('images/', ''))
            for item in self.data
        ]
        self.labels = [item['objects'] for item in self.data]  # Full object information

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image_path = self.image_paths[index]
        objects = self.labels[index]

        # Load image and apply transformations
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Assign binary label: 1 if there is a category other than "red blood cell", 0 otherwise
        has_other_category = any(obj['category'] != 'red blood cell' for obj in objects)
        label = 1 if has_other_category else 0

        return {'X': image, 'label': label, 'indices': [label]}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            the total number of images in the dataset.
        """
        return len(self.image_paths)
