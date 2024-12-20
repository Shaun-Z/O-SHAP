import os
import csv
import numpy as np
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from PIL import Image

class LisaTrafficLightDataset(BaseDataset):
    """
    A dataset class for the LISA traffic light dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(num_classes=2)  # Binary classification: 'go' and 'stop'
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

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.phase == 'train':
            self.data_path = os.path.join(self.dataroot, "dayTrain/dayTrain")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.data_path = os.path.join(self.dataroot, "daySequence1/daySequence1", "frames")
            self.annotation_file = os.path.join(self.dataroot, "Annotations/Annotations", "daySequence1", "frameAnnotationsBOX.csv")
            if not os.path.exists(self.annotation_file):
                raise FileNotFoundError(f"Annotations file not found: {self.annotation_file}")
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

        self.labels = ['other', 'go']

        self.load_data()

    def load_data(self):
        """
        Load data from the dataset and annotations.
        """
        self.image_paths = []
        self.image_labels = []

        if self.phase == 'train':
            for clip in os.listdir(self.data_path):
                clip_path = os.path.join(self.data_path, clip)
                if os.path.isdir(clip_path):
                    annotation_file = os.path.join(self.dataroot, "Annotations", "Annotations", "dayTrain", clip,
                                                   "frameAnnotationsBOX.csv")
                    if not os.path.exists(annotation_file):
                        continue
                    with open(annotation_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter=';')
                        for row in reader:
                            img_name = row['Filename']
                            label = row['Annotation tag']
                            img_path = os.path.join(clip_path, 'frames', os.path.basename(img_name))
                            if os.path.exists(img_path):
                                self.image_paths.append(img_path)
                                self.image_labels.append(self.label_to_index(label))
        else:
            with open(self.annotation_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    img_name = row['Filename']
                    label = row['Annotation tag']
                    img_path = os.path.join(self.data_path, os.path.basename(img_name))
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.image_labels.append(self.label_to_index(label))

    def label_to_index(self, label):
        """
        Convert label name to index.

        Parameters:
            label (str): Label name (e.g., 'go', 'stop')

        Returns:
            int: Corresponding index for the label.
        """
        label_mapping = {'go': 1}
        return label_mapping.get(label.lower(), 0)  # Default to 0 if label is not 'go'

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        img_path = self.image_paths[index]
        label = self.image_labels[index]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {'X': image, 'label': label, 'indices': [label]}  # {image, label, class indices}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            the total number of images in the dataset.
        """
        return len(self.image_paths)
