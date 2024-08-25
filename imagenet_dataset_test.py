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
import torch

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.train_options import TrainOptions
from datasets.imagenet_dataset import ImageNetDataset
opt = TrainOptions().parse()
dataset = ImageNetDataset(opt)
print(dataset[0]['X'].shape, dataset[0]['Y_train'], dataset[0]['Y'])

print(dataset[0]['X'].to(torch.device(opt.gpu_ids))[0][0][0].dtype)

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(dataset[i]['X'], cmap='gray')
#     plt.title(dataset[i]['Y'])
#     plt.axis('off')
# plt.show()