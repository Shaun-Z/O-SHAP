import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, slic

# %% Basic segmentation: Filling 28*28 with 2*2 segments
# def basic_segmentation(img):
#     seg = np.zeros((28,28),dtype = int)
#     for i in range(14):
#         for j in range(14):
#             block_number = i * 14 + j
#             seg[i*2:(i+1)*2,j*2:(j+1)*2] = block_number
#     return seg

class basic_segment:
    def __init__(self, img):
        '''
        img: np.array, shape=(H, W)
        '''
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.factor = H//28 # 224/28=8
        # self.img = img

        Feature_0 = np.zeros((H,W),dtype = int)

        Feature_1 = np.zeros((H,W),dtype = int)
        Feature_1[:,:self.factor*8], Feature_1[:,self.factor*8:W-self.factor*8], Feature_1[:,W-self.factor*8:]= 0,1,2
        
        Feature_2 = np.zeros((H,W),dtype = int)
        H1, H2, H3= self.factor*8, self.factor*12, self.factor*8
        W1, W2, W3= self.factor*8, self.factor*12, self.factor*8
        Feature_2[:H1,:W1], Feature_2[H1:H1+H2,:W1], Feature_2[H1+H2:H1+H2+H3,:W1] = 0, 1, 2
        Feature_2[:H1,W1:W1+W2], Feature_2[H1:H1+H2,W1:W1+W2], Feature_2[H1+H2:H1+H2+H3,W1:W1+W2] = 3, 4, 5 
        Feature_2[:H1,W1+W2:W1+W2+W3], Feature_2[H1:H1+H2,W1+W2:W1+W2+W3], Feature_2[H1+H2:H1+H2+H3,W1+W2:W1+W2+W3] = 6, 7, 8

        Feature_3 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*4):
            for j in range(0, W, self.factor*4):
                Feature_3[i:i+self.factor*4,j:j+self.factor*4] = num
                num += 1

        Feature_4 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*2):
            for j in range(0, W, self.factor*2):
                Feature_4[i:i+self.factor*2,j:j+self.factor*2] = num
                num += 1

        Feature_5 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*1):
            for j in range(0, W, self.factor*1):
                Feature_5[i:i+self.factor*1,j:j+self.factor*1] = num
                num += 1

        self.features_list = [Feature_0, Feature_1, Feature_2, Feature_3, Feature_4, Feature_5]

    def get_mask(self, feature_ID=5):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)
        # print(feature)
        # Display heatmap of basic_seg
        plt.figure(figsize=(18,18))
        plt.imshow(feature, cmap='cool', interpolation='nearest')
        for i in range(self.W):
            for j in range(self.H):
                if i > 0 and feature[i, j] != feature[i-1, j]:
                    plt.plot([j-0.5, j+0.5], [i-0.5, i-0.5], color='black', linewidth=1)
                if j > 0 and feature[i, j] != feature[i, j-1]:
                    plt.plot([j-0.5, j-0.5], [i-0.5, i+0.5], color='black', linewidth=1)
                if i < self.W-1 and feature[i, j] != feature[i+1, j]:
                    plt.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color='black', linewidth=1)
                if j < self.H-1 and feature[i, j] != feature[i, j+1]:
                    plt.plot([j+0.5, j+0.5], [i-0.5, i+0.5], color='black', linewidth=1)

        plt.xticks([0, self.W-1], fontsize=35)
        plt.yticks([0, self.H-1], fontsize=35)
        
        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()

    # def __call__(self):
    #     return self.seg

if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from datasets import create_dataset
    import sys
    basic_seg = basic_segment(np.zeros((28*8,28*8)))
    print(basic_seg.H, basic_seg.W)
    for i in range(6):
        basic_seg.plot_segments(feature_ID=i, savename=f'./basic_seg_layer_{i}.svg')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)
    # plt.title(f"Original Image")
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.imshow(segments_qs)
    # plt.title(f"Quickshift Segmentation")
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.imshow(segments_slic)
    # plt.title(f"SLIC Segmentation")
    # plt.axis('off')
    # plt.show()
