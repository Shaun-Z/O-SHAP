import os
import json
import torch
import numpy as np
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

class MalariaDataset(BaseDataset):
    """
    A dataset class for the malaria dataset, including bounding box information and box map generation.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=2)  # Binary classification: red blood cell (1) vs others (0)
        parser.add_argument('--resize', type=int, nargs=2, default=[224, 224], help='resize the image to this size')
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
        self.transform = transforms.Compose([
            transforms.Resize(self.opt.resize),
            transforms.RandomHorizontalFlip() if self.phase == 'train' else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if self.phase == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean = (-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std = (1 / np.array(self.std)).tolist()
            ),
        ])

        self.box_transform = transforms.Compose([
            transforms.Resize(self.opt.resize),
        ])

        self.labels = ['red blood cell', 'others']

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

        # Extract file paths and objects (with bounding_box)
        self.image_paths = [
            os.path.join(self.images_dir, item['image']['pathname'].lstrip('/').replace('images/', ''))
            for item in self.data
        ]
        self.objects = [item['objects'] for item in self.data]

    def generate_box_map(self, image_size, boxes):
        """
        Generate a box map for the given image and bounding boxes, only for non-red blood cells.

        Parameters:
            image_size (tuple): (width, height) of the image.
            boxes (list): List of bounding boxes.

        Returns:
            box_map (PIL.Image.Image): An RGB image with boxes drawn on it.
        """
        box_map = Image.new('RGBA', image_size, (0, 0, 0, 0))  # Create a black image
        draw = ImageDraw.Draw(box_map)

        # Iterate over bounding boxes and only draw non-red blood cells
        for box in boxes:
            if box['category'] != 'red blood cell':  # Filter non-red blood cells
                coords = [
                    box['bounding_box']['minimum']['c'],  # xmin
                    box['bounding_box']['minimum']['r'],  # ymin
                    box['bounding_box']['maximum']['c'],  # xmax
                    box['bounding_box']['maximum']['r']  # ymax
                ]
                draw.rectangle(coords, outline=(0, 100, 0, 255), width=20)  # Magenta box with thicker width
        return box_map

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image_path = self.image_paths[index]
        objects = self.objects[index]

        # Load the image
        image = Image.open(image_path).convert('RGB')
        image_size = image.size  # Original image size (width, height)

        # Generate the box map for non-red blood cells
        box_map = self.generate_box_map(image_size, objects)

        # Apply transforms
        normalized_image = self.transform(image)
        box_map = self.box_transform(box_map)

        # Convert box_map to tensor
        box_map = transforms.ToTensor()(box_map)

        # Assign binary label: 1 if there is a category other than "red blood cell", 0 otherwise
        has_other_category = any(obj['category'] != 'red blood cell' for obj in objects)
        label = 1 if has_other_category else 0

        return {'X': normalized_image, 'label': label, 'bounding': box_map, 'indices': [label]}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            the total number of images in the dataset.
        """
        return len(self.image_paths)
