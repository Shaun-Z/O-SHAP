import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import shap

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation
from util.color import red_transparent_blue

class ShapExplanation(BaseExplanation):
    
    @staticmethod
    def modify_commandline_options(parser):
        # rewrite default values
        parser.add_argument('--n_evals', type=int, default=5000, help='the number of iterations. The larger the number, the finer the granularity of the significance analysis and the longer the computation consumes time')
        parser.set_defaults(batch_size=50)
        return parser

    def __init__(self, opt):
        super(ShapExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)
        self.explainer = self.define_explainer(self.predict, self.dataset)

    def explain(self, img_index: int):
        X = self.dataset[img_index]['X'].unsqueeze(0)
        Y_class = self.dataset[img_index]['label']
        Y = self.dataset[img_index]['Y']
        Class_list = [self.dataset.label2id[l] for l in Y.split(',')]

        input_img = transform(X)
        
        output_indexes = Class_list if len(self.opt.index_explain)==0 else self.opt.index_explain
        
        self.shap_values = self.explainer(input_img, max_evals=self.opt.n_evals, batch_size=self.opt.batch_size, outputs=output_indexes)
        
        Y = self.dataset[img_index]['Y']
        os.makedirs(f"results/{self.opt.explanation_name}/{self.opt.name}/value", exist_ok=True)
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/value/P{img_index}_{Y}.npy", np.moveaxis(self.shap_values.values[0],-1, 0))

    def predict(self, img: np.ndarray) -> torch.Tensor:
        self.model.input = nhwc_to_nchw(torch.Tensor(img)).to(self.device)
        self.model.forward()
        # y_prob = F.softmax(self.model.output, dim = -1)
        return self.model.output

    def define_explainer(self, pred_fn, dataset):
        Xtr = transform(dataset[0]['X'])
        # out = predict(Xtr[0:1])
        masker_blur = shap.maskers.Image("blur(64, 64)", Xtr.shape)

        explainer = shap.Explainer(pred_fn, masker_blur, output_names=dataset.labels)
        return explainer

    def plot(self, save_path: str = None):
        data = inv_transform(self.shap_values.data).cpu().numpy()[0] # 原图
        values = [val for val in np.moveaxis(self.shap_values.values[0],-1, 0)] # shap值热力图

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shap.image_plot(shap_values=values,
                        pixel_values=data,
                        labels=self.shap_values.output_names,
                        show=False)
            print(f"Saving the image to {save_path}", end='\r')
            plt.savefig(save_path)
            plt.close()
        else:
            shap.image_plot(shap_values=values,
                        pixel_values=data,
                        labels=self.shap_values.output_names)

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform= [
    # transforms.Resize(224),
    # transforms.Lambda(lambda x: x*(1/255)),
    # transforms.Normalize(mean=mean, std=std),
    transforms.Lambda(nchw_to_nhwc),
]

inv_transform= [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Normalize(
        mean = (-1 * np.array(mean) / np.array(std)).tolist(),
        std = (1 / np.array(std)).tolist()
    ),
    transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)