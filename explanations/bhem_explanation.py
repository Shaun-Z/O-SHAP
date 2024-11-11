import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import random

from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

from util.segmentation import basic_segment, hierarchical_segment
from util.color import red_transparent_blue

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation

from itertools import combinations
import tqdm

import math

def safe_division(numerator, denominator):
    try:
        return numerator / denominator
    except OverflowError:
        return float('inf')  # or some other appropriate value

def all_subsets(lst):
    for r in range(len(lst) + 1):
        for subset in combinations(lst, r):
            yield list(subset)

def n_of_all_subsets(lst, n=1):
    total_subsets = 2 ** len(lst)
    n = total_subsets if n > total_subsets else n
    
    subsets = set()
    lst = list(lst)
    while len(subsets) < n:
        subset = []
        for item in lst:
            if random.choice([True, False]):
                subset.append(item)
        subsets.add(tuple(subset))  # 使用 tuple 使其可哈希
    
    return [list(s) for s in subsets]

class layer:
    def __init__(self, image, layer_ID):
        '''
        image: np.ndarray (C, H, W)
        layer_ID: int
        '''
        self.image = image
        self.layer_ID = layer_ID

        # basic_seg = hierarchical_segment(image)
        basic_seg = basic_segment(image)
        seg_func = lambda img: basic_seg.get_mask(feature_ID=layer_ID)
            
        self.seg_func = seg_func    # Segmentation function

        if layer_ID == 0:
            self.segment = np.zeros_like(image[0])
            self.segment_num = 1
            self.masked_image = image
        else:
            self.segment = self.seg_func(image)
            self.segment_num = np.unique(self.segment).shape[0]
            self.masked_image = None

        self.segment_mapping = {}
        for key in np.unique(self.segment):
            self.segment_mapping[key] = np.where(self.segment == key)

        self.seg_active = None
        
    def mask_image(self, seg_keys: list):
        masks = list(map(self.segment_mapping.get, seg_keys))
        img = np.zeros_like(self.image)
        if len(masks) != 0:
            for mask in masks:
                img[:,mask[0],mask[1]] = self.image[:,mask[0],mask[1]]
        self.masked_image = img
        self.seg_active = np.zeros(self.segment_num)
        self.seg_active = seg_keys

    def print_info(self, draw=False):
        assert self.segment is not None
        logging.info(f'''Layer {self.layer_ID}
            layer_ID:           {self.layer_ID}
            segment_num:        {self.segment_num}
            seg_keys:           {self.segment_mapping.keys()}
            segment:            {self.plot_segment() if draw else "Not Draw"}
            seg_active:         {self.seg_active}
            masked_image:       {self.plot_masked_image() if draw else "Not Draw"}
        ''')

    def plot_segment(self):
        assert self.segment is not None
        plt.imshow(self.segment)
        plt.colorbar()
        plt.title("Segment")
        plt.show()
        return "Plot Segment"

    def plot_masked_image(self):
        if self.masked_image is None:
            return None
        else:
            plt.imshow(self.masked_image.permute(1,2,0))
            plt.colorbar()
            plt.title("Masked Image")
            plt.show()
            return "Plot Masked Image"

class BhemExplanation(BaseExplanation):

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--layer_num', type=int, default=4, help='the number of layers')
        parser.add_argument('--approx', action='store_true', help='use approximation (model linearity and feature independence)')
        return parser
    
    def __init__(self, opt):
        super(BhemExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)
        self.layer_num = opt.layer_num+1
        self.approx = opt.approx
        # self.explainer = self.define_explainer(self.predict, self.dataset)

    def initialize_layers(self, image):
        '''
        image: np.ndarray (C, H, W)
        '''
        self.image = image
        self.layers = [layer(image, layer_ID=0)]
        self.layers += [layer(image, layer_ID=i) for i in range(1, self.layer_num)]

        self.mappings = {}
    
        # Initialize mapping between layers
        for i in range(1, self.layer_num-1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            temp_dict = {}
            for key in current_layer.segment_mapping.keys():
                indexes = current_layer.segment_mapping[key]   # Get indexes
                temp_dict[key] = np.unique(next_layer.segment[indexes])
            self.mappings[f'{i}{i+1}'] = temp_dict
        
        for i in range(2, self.layer_num):
            current_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            temp_dict = {}
            for key in current_layer.segment_mapping.keys():
                indexes = current_layer.segment_mapping[key]   # Get indexes
                temp_dict[key] = np.unique(prev_layer.segment[indexes])
            self.mappings[f'{i}{i-1}'] = temp_dict
        
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        self.model.input = img.to(self.device)
        self.model.forward()
        return self.model.output
    
    def get_current_masked_image(self, image, seg_keys: list):
        '''
        image: tensor (C, H, W)
        '''
        assert len(seg_keys) == self.layer_num-1
        img = np.zeros_like(image)
        for i in range(1, self.layer_num):
            self.layers[i].mask_image(seg_keys[i-1])
            img += self.layers[i].masked_image
        return torch.Tensor(img)

    def explain(self, img_index: int):
        input_img = self.dataset[img_index]['X']    # Get input image (C, H, W)
        self.Y = self.dataset[img_index]['Y']
        Y_class = self.dataset[img_index]['label']
        self.class_list = [self.dataset.label2id[l] for l in self.Y.split(',')]
        self.initialize_layers(input_img)   # Initialize layers. The class will have the following attributes: layers, mappings. Each layer will have the following attributes: segment, segment_num, masked_image, seg_active, segment_mapping
        self.print_explanation_info()

        indexes = [list(self.layers[i].segment_mapping.keys()) for i in range(1, self.layer_num)]
        # scores = np.zeros((1, len(self.dataset.labels), int(input_img.shape[-1]*input_img.shape[-2]/16/16)))
        scores = np.zeros((1, len(self.dataset.labels), int(input_img.shape[-2]), input_img.shape[-1]))

        # block_num = (2**(len(indexes[0])+len(indexes[1])+len(indexes[2])+len(indexes[3])))

        cnt = 0
        for f1 in indexes[0]:
            f1s = indexes[0].copy() # copy
            f1s.remove(f1)
            f2_idx = self.mappings['12'][f1]

            for f2 in f2_idx:
                f2s =list(f2_idx.copy()) # copy
                f2s.remove(f2)
                f3_idx = self.mappings['23'][f2]

                # for f3 in tqdm.tqdm(f3_idx, desc="Layer 3"):
                for f3 in f3_idx:
                    f3s = list(f3_idx.copy()) # copy
                    f3s.remove(f3)
                    f4_idx = self.mappings['34'][f3]

                    # for f4 in tqdm.tqdm(f4_idx, desc="Layer 4"):
                    for f4 in f4_idx:

                        total_num = len(indexes[0])+len(f2_idx)+len(f3_idx)+len(f4_idx)

                        f4s = list(f4_idx.copy())

                        f4s.remove(f4)            
                        
                        # img1 = self.get_current_masked_image(input_img, [f1s, [], [], []]) * math.factorial(len(indexes[0])) * math.factorial(total_num - len(indexes[0])-1) / math.factorial(total_num) \
                        #      + self.get_current_masked_image(input_img, [[], f2s, [], []]) * math.factorial(len(f2_idx)) * math.factorial(total_num - len(f2_idx)-1) / math.factorial(total_num) \
                        #      + self.get_current_masked_image(input_img, [[], [], f3s, []]) * math.factorial(len(f3_idx)) * math.factorial(total_num - len(f3_idx)-1) / math.factorial(total_num) \
                        #      + self.get_current_masked_image(input_img, [[], [], [], f4s]) * math.factorial(len(f4_idx)) * math.factorial(total_num - len(f4_idx)-1) / math.factorial(total_num)

                        if self.approx:
                            img1 = self.get_current_masked_image(input_img, [f1s, [], [], []]) / safe_division(2**(len(indexes[0])-1), 1) \
                                + self.get_current_masked_image(input_img, [[], f2s, [], []]) / safe_division(2**(len(f2_idx)-1), 1) \
                                + self.get_current_masked_image(input_img, [[], [], f3s, []]) / safe_division(2**(len(f3_idx)-1), 1) \
                                + self.get_current_masked_image(input_img, [[], [], [], f4s]) / safe_division(2**(len(f4_idx)-1), 1)

                            img = img1 + self.get_current_masked_image(input_img, [[], [], [], [f4]])

                            '''plt.figure(figsize=(20, 10))
                            plt.subplot(1, 2, 1)
                            plt.imshow(self.dataset.inv_transform(img).permute(1,2,0), vmin=0, vmax=255)
                            # plt.imshow(img.permute(1,2,0), vmin=0, vmax=255)
                            plt.title("Include f4")
                            plt.colorbar()
                            plt.subplot(1, 2, 2)
                            plt.imshow(self.dataset.inv_transform(img1).permute(1,2,0), vmin=0, vmax=255)
                            # plt.imshow(img1.permute(1,2,0), vmin=0, vmax=255)
                            plt.colorbar()
                            plt.title("Exclude f4")
                            plt.savefig(f'./img_res/img({f1})({f2})({f3})({f4}).png')
                            plt.close()'''
                            P1 = self.predict(img.unsqueeze(0))
                            P2 = self.predict(img1.unsqueeze(0))
                            mask = self.layers[4].segment_mapping.get(f4)
                            if mask is not None:
                                for x,y in zip(mask[0], mask[1]):
                                    scores[:,:,x,y] += np.array((P1-P2).cpu().detach().numpy())/len(mask[0])
                        # feature_group_num = (2**(len(f1s)+len(f2s)+len(f3s)+len(f4s)))
                        else:
                            img1 = torch.zeros_like(input_img)
                            img = torch.zeros_like(input_img)
                            s1,s2,s3,s4 = 0,0,0,0
                            # print(f"{len(f1s)}\t{len(f2s)}\t{len(f3s)}\t{len(f4s)}")
                            for subset1 in n_of_all_subsets(f1s):
                                for subset2 in n_of_all_subsets(f2s):
                                    for subset3 in n_of_all_subsets(f3s):
                                        for subset4 in n_of_all_subsets(f4s):
                                            # print(subset4)
                                            cnt+=1
                                            # print(f"{cnt/total_num}", end='\r')

                                            img1 += self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])   # Get masked image without f4 (C, H, W) tensor
                                            subset4.append(f4)

                                            img += self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])    # Get masked image with f4 (C, H, W) tensor

                                            # plt.figure(figsize=(20, 10))
                                            # plt.subplot(1, 2, 1)
                                            # plt.imshow(self.dataset.inv_transform(img).permute(1,2,0), vmin=0, vmax=255)
                                            # plt.title("Include f4")
                                            # plt.colorbar()
                                            # plt.subplot(1, 2, 2)
                                            # plt.imshow(self.dataset.inv_transform(img1).permute(1,2,0), vmin=0, vmax=255)
                                            # plt.colorbar()
                                            # plt.title("Exclude f4")
                                            # plt.savefig(f'./img_res/img({f1}_{s1})({f2}_{s2})({f3}_{s3})({f4}_{s4}).png')
                                            # plt.close()

                                            P1 = self.predict(img.unsqueeze(0))
                                            P2 = self.predict(img1.unsqueeze(0))
                                            # P1.shape, P2.shape: (1,200)

                                            mask = self.layers[4].segment_mapping.get(f4)
                                            if mask is not None:
                                                for x,y in zip(mask[0], mask[1]):
                                                    scores[:,:,x,y] += np.array((P1-P2).cpu().detach().numpy())/(len(subset1)+len(subset2)+len(subset3)+len(subset4))
                                            s4 +=1
                                        s3 +=1
                                    s2 +=1
                                s1 +=1
                        
                        # np.save('img1.npy', img1/feature_group_num)
                        # np.save('img.npy', img/feature_group_num)
                        # np.save('input_img.npy', input_img)
                        # print(cnt)

                        # img /= feature_group_num
                        # img1 /= feature_group_num

        self.scores = scores.reshape(len(self.dataset.labels), input_img.shape[-2], input_img.shape[-1])

        self.scores = np.expand_dims(self.scores[self.class_list], axis=-1) # Add channel dimension

        # zoom_factors = (1, 16, 16, 1)  # (N 维度不变，宽高维度放大 16 倍，通道维度不变)

        # self.scores_to_save = zoom(self.scores, zoom_factors, order=0)/16/16
        self.scores_to_save = self.scores

        os.makedirs(f'results/{self.opt.explanation_name}/{self.opt.name}/value', exist_ok=True)
        np.save(f'results/{self.opt.explanation_name}/{self.opt.name}/value/P{img_index}_{self.Y}.npy', self.scores_to_save)
        # print(cnt)
        return self.scores
                
    def plot(self, save_path: str = None):
        image = self.dataset.inv_transform(self.image).permute(1,2,0)
        image_show = image.mean(axis=-1)

        result_show = self.scores_to_save.sum(axis=-1)    # (N, 224, 224, 1)

        labels_to_display = [self.dataset.labels[index] for index in self.class_list]

        fig, axes = plt.subplots(nrows=1, ncols=result_show.shape[0]+1, figsize=(8,6), squeeze=False)
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        max_val = np.nanpercentile(np.abs(result_show), 99.9)
        for i in range(result_show.shape[0]):
            axes[0, i+1].set_title(labels_to_display[i])
            axes[0, i+1].imshow(image_show, cmap=plt.get_cmap('gray'), alpha=0.3)
            axes[0, i+1].imshow(result_show[i], cmap=red_transparent_blue, vmin=-max_val,vmax=max_val)
            axes[0, i+1].axis('off')
            im = axes[0, i+1].imshow(result_show[i], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)

        cb = plt.colorbar(im, ax=np.ravel(axes).tolist(), label="BHEM value", orientation="horizontal", aspect=30)
        cb.outline.set_visible(False)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # print(f"Saving the image to {save_path}", end='\r')
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def print_explanation_info(self):
        logging.info(f'''Explanaion Info:
            layer_num:          {self.layer_num-1}
            layers_mapping:     {self.mappings.keys()}
        ''')
        for layer_ID in range(1, self.layer_num):
            self.layers[layer_ID].print_info(draw=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from options.explain_options import ExplainOptions
    from explanations import create_explanation

    opt = ExplainOptions().parse()   # get explain options
    explainer = create_explanation(opt)
    explainer.explain(1)
    explainer.plot()

'''
python explain.py -d ./data/tiny-imagenet -n Resnet50onImageNet -g 0 -m res_class --net_name resnet50 --dataset_name imagenet --eval
--epoch 15
'''