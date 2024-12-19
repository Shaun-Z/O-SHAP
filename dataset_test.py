'''
To test this script, run the following command:
----------------
python imagenet_dataset_test.py --dataroot ./data/tiny-imagenet --gpu_ids -1
----------------
or
----------------
python imagenet_dataset_test.py -d ./data/tiny-imagenet -g -1
----------------
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.imagenet_dataset import ImageNetDataset

from datasets import create_dataset

if __name__ == '__main__':
    
    opt = TestOptions().parse()
    dataset = create_dataset(opt)

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"X:{data['X'].shape}\tlabel:{data['label']}\tindices:{data['indices']}")

    print(dataset.labels)
