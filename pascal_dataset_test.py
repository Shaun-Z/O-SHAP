'''
To test this script, run the following command:
----------------
python pascal_dataset_test.py --dataroot ./data/pascal_voc_2012 --gpu_ids -1 --dataset_name pascalvoc --phase val --batch_size 1 --segmentation
----------------
or
----------------
python pascal_dataset_test.py -d ./data/pascal_voc_2012 -g -1 --dataset_name pascalvoc --phase val --batch_size 1 --segmentation
----------------
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets import create_dataloader

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    # dataset = PascalVocDataset(opt)
    dataloader = create_dataloader(opt)
    dataset = dataloader.dataset

    # for i, data in enumerate(dataloader):
    #     if opt.segmentation:
    #         print(data["X"].shape, data['Y'], data['mask'])
    #     else:
    #         print(data["X"].shape, data['Y'])
    

    for i in range(len(dataset)):
        data = dataset[i]
        Y = data['Y']
        Y_class = data['Y_class']
        mask = data['mask']
        unique_indices = np.unique((np.array(mask)*255).astype(np.uint8))[1:-1]-1
        print(f"X.shape:{data['X'].shape}\tY:{data['Y']}\t{dataset.get_class_list(Y_class)}\tunique_indices:{unique_indices}\tmask:{(mask*255).unique()}")
        # print(dataset.X[i])
        # print(dataset.mask[i])

    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     Y = data['Y']
    #     print(f"X.shape:{data['X'].shape}\tY_class:{data['Y_class']}\tY:{data['Y']}")

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(dataset.inv_transform(data['X']).permute(1,2,0))
        # plt.subplot(1, 2, 2)
        # plt.imshow(data['X'].permute(1,2,0))
        # plt.axis('off')
        # plt.show()

        # break
        # if Y == dataset.labels[index]:
        #     # print(dataset[i]['X'].shape)
        #     print(f'{i}\t{Y}\t{index}')
        # else:
        #     print('Error')
        #     exit()

