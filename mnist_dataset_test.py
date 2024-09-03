'''
To test this script, run the following command:
----------------
python mnist_dataset_test.py --dataroot ./datasets/mnist --gpu_ids -1
----------------
or
----------------
python mnist_dataset_test.py -d ./datasets/mnist -g -1
----------------
'''

import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from datasets.mnist_dataset import MnistDataset

if __name__ == '__main__':

    opt = TrainOptions().parse()
    dataset = MnistDataset(opt)
    print(dataset[0]['X'].shape, dataset[0]['Y'])

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(dataset[i]['X'], cmap='gray')
        plt.title(dataset[i]['Y'])
        plt.axis('off')
    plt.show()

    print(len(dataset))