import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

import torch

from util.segmentation import basic_segment
from util.color import red_transparent_blue

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation

from itertools import combinations
import tqdm

# RC Viz Code

def all_subsets(lst):
    for r in range(len(lst) + 1):
        for subset in combinations(lst, r):
            yield list(subset)

class layer:
    def __init__(self, image, layer_ID):
        '''
        image: np.ndarray (C, H, W)
        layer_ID: int
        '''
        self.image = image
        self.layer_ID = layer_ID

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
        return parser
    
    def __init__(self, opt):
        super(BhemExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)
        self.layer_num = opt.layer_num+1
        # self.explainer = self.define_explainer(self.predict, self.dataset)

    def initialize_layers(self, image):
        '''
        image: np.ndarray (C, H, W)
        '''
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

    # def define_explainer(self, pred_fn, dataset):
    #     self.pred_fn = pred_fn
    #     return explainer
    
    # def explain(self, img_index: int):
    #     X = self.dataset[img_index]['X'].unsqueeze(0)
    #     input_img = X
    #     self.shap_values = self.explainer(input_img, outputs=self.opt.index_explain)
    
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
        Y = self.dataset[img_index]['Y']
        self.initialize_layers(input_img)   # Initialize layers. The class will have the following attributes: layers, mappings. Each layer will have the following attributes: segment, segment_num, masked_image, seg_active, segment_mapping
        self.print_explanation_info()

        indexes = [list(self.layers[i].segment_mapping.keys()) for i in range(1, self.layer_num)]
        scores = np.zeros((1, len(self.dataset.labels), int(input_img.shape[-1]*input_img.shape[-2]/16/16)))

        block_num = (2**(len(indexes[0])+len(indexes[1])+len(indexes[2])+len(indexes[3])))

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
                        f4s = list(f4_idx.copy())
                        f4s.remove(f4)
                            
                        s1,s2,s3,s4 = 0,0,0,0

                        feature_group_num = (2**(len(f1s)+len(f2s)+len(f3s)+len(f4s)))
                        total_num = feature_group_num*block_num
                        
                        img1 = torch.zeros_like(input_img)
                        img = torch.zeros_like(input_img)

                        for subset1 in all_subsets(f1s):
                            for subset2 in all_subsets(f2s):
                                for subset3 in all_subsets(f3s):
                                    for subset4 in all_subsets(f4s):
                                        # print(f"Feature 4: {subset4}")
                                        cnt+=1
                                        print(f"{cnt/total_num}", end='\r')

                                        img1 += self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])   # Get masked image without f4 (C, H, W) tensor
                                        subset4.append(f4)

                                        img += self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])    # Get masked image with f4 (C, H, W) tensor

                                        # img1 = self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])   # Get masked image without f4 (C, H, W) tensor
                                        # subset4.append(f4)

                                        # img = self.get_current_masked_image(input_img, [subset1, subset2, subset3, subset4])    # Get masked image with f4 (C, H, W) tensor

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

                                        # P1 = self.predict(img.unsqueeze(0))
                                        # P2 = self.predict(img1.unsqueeze(0))
                                        # P1.shape, P2.shape: (1,200)

                                        # scores[:,:,f4] += np.array((P1-P2).cpu().detach().numpy())/feature_group_num
                                        s4 +=1
                                    s3 +=1
                                s2 +=1
                            s1 +=1

                        img /= feature_group_num
                        img1 /= feature_group_num
                        P1 = self.predict(img.unsqueeze(0))
                        P2 = self.predict(img1.unsqueeze(0))
                        scores[:,:,f4] += np.array((P1-P2).cpu().detach().numpy())
        self.scores = scores.reshape(1, len(self.dataset.labels), int(input_img.shape[-1]/16), int(input_img.shape[-2]/16))
        np.save(f'results/{self.opt.explanation_name}/{self.opt.name}/P{img_index}_{Y}.npy', self.scores)
        return scores
                
    def plot(self, img_index: int, save_path: str = None):
        # result = scores.reshape(1,10, 14, 14)
        input_img = self.dataset[img_index]['X']    # Get input image (C, H, W)
        image = self.dataset.inv_transform(input_img).permute(1,2,0)
        Y = self.dataset[img_index]['Y']
        exp_result = np.load(f'results/{self.opt.explanation_name}/{self.opt.name}/P{img_index}_{Y}.npy')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40,10), squeeze=False)

        axes[0, 0].imshow(image, alpha=0.3)
        axes[0][0].axis('off')
        max_val = np.nanpercentile(exp_result[0], 99.9)
        for i in range(10):
            axes[0][i+1].imshow(self.image, alpha=0.3)
            axes[0][i+1].imshow(exp_result[0][i], cmap=red_transparent_blue, vmin=-np.nanpercentile(exp_result[0], 99.9),vmax=np.nanpercentile(exp_result[0], 99.9))
            axes[0][i+1].axis('off')
            im = axes[0, i+1].imshow(exp_result[0][i], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)

        plt.colorbar( im, ax=np.ravel(axes).tolist(), label="BHEM value", orientation="horizontal", aspect=40 / 0.2)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_single_aopc_value(self, img_index: int, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        model = self.predict
        image = self.dataset[img_index]['X'].unsqueeze(0)   # get the base image CxHxW
        Y_class = self.dataset[img_index]['Y_class']
        X_pred = model(image)
        base_value = X_pred.softmax(dim=-1).flatten()[Y_class].item()
        Y = self.dataset[img_index]['Y']

        AOPC = np.array([0.0]*len(percents))
        for i in range(len(percents)):
            res = self.delete_top_k_features(percents[i], img_index).unsqueeze(0)
            AOPC[i] = base_value - model(res).softmax(dim=-1).flatten()[Y_class]
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/aopc{img_index}_{Y}.npy", AOPC)
        return AOPC
    
    def delete_top_k_features(self, k, img_index: int):
        img = self.dataset[img_index]['X']  # get the image CxHxW
        Y = self.dataset[img_index]['Y']
        path_to_value = f"results/{self.opt.explanation_name}/{self.opt.name}/P{img_index}_{Y}.npy"    # 1 x classes x H x W

        value = np.load(path_to_value)
        total_sum = np.sum(value)

        # Sort the values in descending order and get the corresponding indices
        sorted_indices = np.argsort(-value.flatten())

        index = int(k * len(sorted_indices))

        # Set the values after the index to 0
        result = img.flatten()
        result[sorted_indices[:index]] = 0
        result = result.reshape(img.shape)

        return result

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