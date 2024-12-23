import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from .base_dataset import BaseDataset


class CelebADataset(BaseDataset):
    """
    A dataset class for the CelebA dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=40)  # Number of labels in the dataset
        return parser

    def __init__(self, opt):
        """
        Initialize the dataset with options and load data.
        """
        super(CelebADataset, self).__init__(opt)
        self.data_root = opt.dataroot  # Use the correct attribute name here
        self.image_dir = os.path.join(self.data_root, "img_align_celeba/img_align_celeba")
        self.label_file = os.path.join(self.data_root, "list_attr_celeba.csv")

        # Load the attribute labels
        with open(self.label_file, "r") as f:
            lines = f.readlines()
            self.labels = lines[0].strip().split(",")[1:]  # Extract label names from the header row
            self.data = [line.strip().split(",") for line in lines[1:]]  # Load data starting from the second line

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize all images to 128x128
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve an item at a specific index.
        """
        # Get the image path and labels
        img_name, *labels = self.data[index]
        img_path = os.path.join(self.image_dir, img_name)
        labels = np.array(labels, dtype=int)

        # Convert -1 to 0 for binary labels
        binary_labels = (labels == 1).astype(np.int32)

        # Find indices of labels that are 1
        indices = np.where(binary_labels == 1)[0].tolist()

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Return a dictionary of image, labels, and indices
        return {
            "X": image,
            "label": torch.tensor(binary_labels, dtype=torch.float32),
            "indices": indices
        }
