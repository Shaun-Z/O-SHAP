import os
import sys
import numpy as np
from datasets.base_dataset import BaseDataset, get_transform
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from glob import glob
import random
import torch


transform = transforms.Compose([transforms.ToTensor(), ])
means = torch.zeros(3)  # RGB三个通道
stds = torch.zeros(3)
num_images = 0
# means = torch.zeros(3)
# stds = torch.zeros(3)
for filename in os.listdir(dir):
    if filename.endswith(('.jpg')):  # 确保只处理图像文件
        img_path = os.path.join(dir, filename)
        image = Image.open(img_path).convert('RGB')  # 确保是 RGB 格式
        image_tensor = transform(image)

        means += torch.mean(image_tensor, dim=(1, 2))
        stds += torch.std(image_tensor, dim=(1, 2))
        num_images += 1

means /= num_images
stds /= num_images
# 输出均值和标准差
means_list = means.tolist()
stds_list = stds.tolist()
self.means = means_list
self.stds = stds_list
print(f'Calculated means: {self.means}')
print(f'Calculated stds: {self.stds}')