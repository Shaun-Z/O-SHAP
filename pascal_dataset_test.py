'''
To test this script, run the following command:
----------------
python pascal_dataset_test.py --dataroot ./data/pascal_voc_2007 --gpu_ids -1
----------------
or
----------------
python pascal_dataset_test.py -d ./data/pascal_voc_2007 -g -1
----------------
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.pascal_dataset import PascalVoc2007

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    dataset = PascalVoc2007(opt)

    for i in range(len(dataset)):
        data = dataset[i]
        Y = data['Y']
        print(f"X:{data['X']}\t X.shape:{data['X'].shape}\tY:{data['Y']}")
        break
        # if Y == dataset.labels[index]:
        #     # print(dataset[i]['X'].shape)
        #     print(f'{i}\t{Y}\t{index}')
        # else:
        #     print('Error')
        #     exit()

    print(len(dataset.train), len(dataset.val), len(dataset.test), len(dataset.classes))

    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(dataset[i]['X'].permute(1,2,0), cmap='gray')
    #     plt.title(dataset.labels_meaning[dataset[i]['Y']])
    #     plt.axis('off')
    # plt.show()