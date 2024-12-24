import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split  # For splitting data
from .base_dataset import BaseDataset
import pandas as pd


class CelebADataset(BaseDataset):
    """
    A dataset class for the CelebA dataset with train-test split.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=40)  # Number of labels in the dataset
        return parser

    def __init__(self, opt):
        """
        Initialize the dataset with options and load data.
        Args:
            opt: The options containing dataroot and other configurations.
        """
        super(CelebADataset, self).__init__(opt)
        self.data_root = opt.dataroot
        self.image_dir = os.path.join(self.data_root, "img_align_celeba/img_align_celeba")
        self.label_file = os.path.join(self.data_root, "list_attr_celeba.csv")
        self.phase = opt.phase

        # Load the attribute labels
        df = pd.read_csv(self.label_file)
        self.labels = df.columns[1:]
        all_data = df.values.tolist()

        train_data, test_data = train_test_split(all_data, test_size=0.3, random_state=42)
        self.data = train_data if self.phase == 'train' else test_data

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        try:
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
            if self.transform is not None:
                image = self.transform(image)

            # Return only "X" and "label" during training
            if self.phase == 'train':
                return {
                    "X": image,
                    "label": torch.tensor(binary_labels, dtype=torch.float32)
                }

            # Include "indices" during testing
            return {
                "X": image,
                "label": torch.tensor(binary_labels, dtype=torch.float32),
                "indices": indices
            }
        except Exception as e:
            # Log the error and skip the current index
            print(f"Error loading data at index {index}: {e}")
            return None
