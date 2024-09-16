'''
To test this script, run the following command:
----------------
python pascal_dataset_test.py --dataroot ./data/pascal_voc_2007 --gpu_ids -1 --phase val
----------------
or
----------------
python pascal_dataset_test.py -d ./data/pascal_voc_2007 -g -1 --phase val
----------------
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.pascalvoc_dataset import PascalVocDataset

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    dataset = PascalVocDataset(opt)

    print(len(dataset.train), len(dataset.val), len(dataset.test), len(dataset.classes))

    for i in range(len(dataset)):
        data = dataset[i]
        Y = data['Y']
        print(f"X.shape:{data['X'].shape}\tY_class:{data['Y_class']}\tY:{data['Y']}\tY_dict:{data['Y_dict']}")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(dataset.inv_transform(data['X']).permute(1,2,0))
        plt.subplot(1, 2, 2)
        plt.imshow(data['X'].permute(1,2,0))
        plt.axis('off')
        plt.show()

        break
        # if Y == dataset.labels[index]:
        #     # print(dataset[i]['X'].shape)
        #     print(f'{i}\t{Y}\t{index}')
        # else:
        #     print('Error')
        #     exit()

