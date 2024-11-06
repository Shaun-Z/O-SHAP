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
from datasets.sgcc_dataset import SgccDataset

if __name__ == '__main__':

    opt = TrainOptions().parse()
    dataset = SgccDataset(opt)
