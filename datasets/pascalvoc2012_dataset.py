import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset, get_transform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import torchvision.transforms as transforms
import torch
from xml.dom.minidom import parse

class PascalVoc2012Dataset(BaseDataset):
    """A dataset class for pascal_voc_2012 dataset.

    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segmentation', action='store_true', help='if specified, load the segmentation dataset')
        parser.set_defaults(num_classes=20)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)  # call the default constructor of BaseDataset
        self.phase = opt.phase # get the phase: train, val, test
        self.dataroot = os.path.join(opt.dataroot)
        data_dir = os.path.join(self.dataroot, "trainval/VOCdevkit/VOC2012")
        self.anno_dir = os.path.join(data_dir, "Annotations")
        self.image_dir = os.path.join(data_dir, "JPEGImages")
        self.num_classes = opt.num_classes
        self.segmentation = opt.segmentation

        self.labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        if self.segmentation:
            self.train_txt = os.path.join(data_dir, "ImageSets/Segmentation/train.txt")
            self.val_txt = os.path.join(data_dir, "ImageSets/Segmentation/val.txt")
        else:
            self.train_txt = os.path.join(data_dir, "ImageSets/Main/train.txt")
            self.val_txt = os.path.join(data_dir, "ImageSets/Main/val.txt")

        self.train_imgIDs = [t.strip() for t in open(self.train_txt)]
        self.val_imgIDs = [t.strip() for t in open(self.val_txt)]

        self.data = []
        self.mask = []

        self.label2id = {label:i for i, label in enumerate(self.labels)}

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.phase == 'train':
            self.transform_mask = transforms.Compose([  # transform for the mask
                transforms.Resize(224),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224, padding = 10),
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([   # transform for the image
                self.transform_mask,
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform_mask = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),     # Please note that some mask values may be cropped out !!!
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([
                self.transform_mask,
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean = (-1 * np.array(self.mean) / np.array(self.std)).tolist(),
                std = (1 / np.array(self.std)).tolist()
            ),
        ])
                
        self.load_data()
        # self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

    def load_data(self):
        """Load data into memory.(here we only load the path to the data cuz the dataset is too large)"""
        # load data from /path/to/data/

        if self.phase == 'train':
            label_set = set()
            
            for img_id in self.train_imgIDs:
                xml_path = os.path.join(self.anno_dir, f'{img_id}.xml')
                dom_tree = parse(xml_path)
                root = dom_tree.documentElement
                objects = root.getElementsByTagName('object')
                labels = set()
                for obj in objects:
                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                        continue
                    tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                    labels.add(tag)
                    label_set.add(tag)
                image_path = os.path.join(self.image_dir, f'{img_id}.jpg')
                if not os.path.exists(image_path):
                    raise Exception(f'file {image_path} not found!')
                self.data.append([image_path, ','.join(list(labels))])
                if self.segmentation:
                    mask_path = os.path.join(self.dataroot, f'trainval/VOCdevkit/VOC2012/SegmentationClass/{img_id}.png')
                    if not os.path.exists(mask_path):
                        raise Exception(f'file {mask_path} not found!')
                    self.mask.append(mask_path)

        elif self.phase == 'val':
            for img_id in self.val_imgIDs:
                xml_path = os.path.join(self.anno_dir, f'{img_id}.xml')
                dom_tree = parse(xml_path)
                root = dom_tree.documentElement
                objects = root.getElementsByTagName('object')
                labels = set()
                for obj in objects:
                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                        continue
                    tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                    labels.add(tag)
                image_path = os.path.join(self.image_dir, f'{img_id}.jpg')
                if not os.path.exists(image_path):
                    raise Exception(f'file {image_path} not found!')
                self.data.append([image_path, ','.join(list(labels))])
                if self.segmentation:
                    mask_path = os.path.join(self.dataroot, f'trainval/VOCdevkit/VOC2012/SegmentationClass/{img_id}.png')
                    if not os.path.exists(mask_path):
                        raise Exception(f'file {mask_path} not found!')
                    self.mask.append(mask_path)
        else:
            raise ValueError(f'Invalid phase: {self.phase}')

    '''
    Reminder:
    Don't call this function from dataloader.dataset.get_class_list(Y_class), because dataloader adds a dimention to the 1st dimension of the tensor.
    '''
    def get_class_list(self, Y_class: torch.Tensor):
        assert isinstance(Y_class, torch.Tensor), "Y_class should be a tensor"
        class_list = torch.nonzero(Y_class).squeeze().tolist()
        if not isinstance(class_list, list):   
            class_list = [class_list]
        return class_list

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image_path, image_label = self.data[index]
        label_indices = [self.label2id[l] for l in image_label.split(',')]
        
        image_data = Image.open(image_path).convert("RGB") # read the image

        x = self.transform(image_data)
        y = np.zeros(self.num_classes).astype(np.float32)
        y[label_indices] = 1.0

        if self.segmentation:   # return the [image], its [class] and the [mask]
            mask = Image.open(self.mask[index])
            mask = self.transform_mask(mask)
            return {'X': x, 'Y_class': y, 'Y': image_label, 'mask': mask}
        else:   # return the [image] and its [class]
            return {'X': x, 'Y_class': y, 'Y': image_label} # return the image and the first class

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)