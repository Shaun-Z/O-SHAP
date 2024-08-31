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

opt = TrainOptions().parse()
dataset = ImageNetDataset(opt)

print(dataset[0]['X'].max(), dataset[0]['X'].min())
print(dataset[2]['X'].max(), dataset[2]['X'].min())
print(dataset[3]['X'].max(), dataset[3]['X'].min())
print(dataset[4]['X'].max(), dataset[4]['X'].min())
print(dataset[5]['X'].max(), dataset[5]['X'].min())

for i in range(len(dataset)):
    data = dataset[i]
    Y = data['Y']
    print(f"X:{dataset.X[i]}\tY_class:{data['Y_class'].numpy()}\tY:{Y}")
    exit()
    # if Y == dataset.labels[index]:
    #     # print(dataset[i]['X'].shape)
    #     print(f'{i}\t{Y}\t{index}')
    # else:
    #     print('Error')
    #     exit()

print(len(dataset.labels), len(dataset.labels_meaning))

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(dataset[i]['X'].permute(1,2,0), cmap='gray')
#     plt.title(dataset.labels_meaning[dataset[i]['Y']])
#     plt.axis('off')
# plt.show()