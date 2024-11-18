import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, slic
from skimage import segmentation, graph
import torch

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

class hierarchical_segment:
    def __init__(self, img, n_segments=10):
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.img = img

        img = self.__preprocess_image(img)
        # 初始过分割
        segments = self.initial_segmentation(img, n_segments)
        # 层次化区域合并
        self.features_list = self.hierarchical_segmentation(img, segments, num_levels=4)
        self.features_list.append(np.zeros((H,W),dtype = int))
        self.features_list.reverse()

    
    def __preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # 转换图像维度
        img = np.transpose(img, (1, 2, 0))
        # 如果图像是浮点型，确保在0-1之间
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)
        else:
            img = img / 255.0
        return img
    
    def initial_segmentation(self, img, n_segments=196):
        # 初始过分割，使用 SLIC 超像素
        segments = segmentation.slic(img, n_segments=n_segments, compactness=10, start_label=0)
        return segments
    
    def __build_rag(self, img, segments):
        segments = segments.astype(int)
        # 构建区域邻接图（RAG）
        rag = graph.rag_mean_color(img, segments, mode='similarity')
        return rag
    
    def hierarchical_segmentation(self, img, segments, num_levels=4):
        # 记录每一层的分割结果
        segments_list = [segments]
        current_segments = segments.copy()

        for level in range(1, num_levels):
            if np.unique(current_segments).shape[0] == 1:
                pass
            else:
                # 在每次迭代中重新构建 RAG
                rag = self.__build_rag(img, current_segments)

                # 计算合并阈值，阈值逐层增大，导致合并的区域增多
                thresh = 15 * level * np.percentile([data['weight'] for u, v, data in rag.edges(data=True)], 98)

                # 合并相似区域
                current_segments = graph.merge_hierarchical(
                                    current_segments, rag, thresh=thresh, rag_copy=False,
                                    in_place_merge=False, merge_func=self.__merge_mean_color,
                                    weight_func=self.__weight_mean_color)

            segments_list.append(current_segments.copy())

        return segments_list

    def __merge_mean_color(self, graph_, src, dst):
        # 定义合并后的区域属性更新方式
        graph_.nodes[dst]['total color'] += graph_.nodes[src]['total color']
        graph_.nodes[dst]['pixel count'] += graph_.nodes[src]['pixel count']
        graph_.nodes[dst]['mean color'] = (graph_.nodes[dst]['total color'] /
                                        graph_.nodes[dst]['pixel count'])

    def __weight_mean_color(self, graph_, src, dst, n):
        # 定义区域之间的权重计算方式
        diff = graph_.nodes[dst]['mean color']*255 - graph_.nodes[n]['mean color']*255
        diff = np.linalg.norm(diff)
        return {'weight': diff}
    
    def __display_segments(self, img, segments, title):
        # 绘制分割边缘
        boundary = segmentation.mark_boundaries(img, segments)
        plt.figure(figsize=(8, 8))
        plt.imshow(boundary)
        plt.title(title)
        plt.axis('off')

    def get_mask(self, feature_ID=5):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)

        self.__display_segments(np.transpose(self.img.detach().cpu().numpy(), (1, 2, 0)), feature, f'Hierarchical Segmentation - Level {feature_ID}')

        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()

