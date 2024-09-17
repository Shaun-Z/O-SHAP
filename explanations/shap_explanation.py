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
        input_img = transform(X)
        self.shap_values = self.explainer(input_img, max_evals=self.opt.n_evals, batch_size=self.opt.batch_size, outputs=self.opt.index_explain)
        Y_class = self.dataset[img_index]['Y_class']
        Y = self.dataset[img_index]['Y']
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/P{img_index}_{Y}.npy", np.moveaxis(self.shap_values.values[0],-1, 0))

    def predict(self, img: np.ndarray) -> torch.Tensor:
        self.model.input = nhwc_to_nchw(torch.Tensor(img)).to(self.device)
        self.model.forward()
        # y_prob = F.softmax(self.model.output, dim = -1)
        return self.model.output

    def define_explainer(self, pred_fn, dataset):
        Xtr = transform(dataset[0]['X'])
        # out = predict(Xtr[0:1])
        masker_blur = shap.maskers.Image("blur(64, 64)", Xtr.shape)

        explainer = shap.Explainer(pred_fn, masker_blur, output_names=None)
        return explainer

    def plot(self, save_path: str = None):
        data = inv_transform(self.shap_values.data).cpu().numpy()[0] # 原图
        values = [val for val in np.moveaxis(self.shap_values.values[0],-1, 0)] # shap值热力图
        shap.image_plot(shap_values=values,
                        pixel_values=data,
                        labels=None,
                        show=False)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(data)
        # plt.title("Original Image")
        # plt.subplot(1, 2, 2)
        # plt.imshow(data, alpha=0.3)
        # plt.imshow(values[0], cmap=red_transparent_blue, vmin=-np.nanpercentile(values, 99.9),vmax=np.nanpercentile(values, 99.9))
        # plt.title("Shap Image")
        if save_path:
            shap.image_plot(shap_values=values,
                        pixel_values=data,
                        labels=None,
                        show=False)
            plt.savefig(save_path)
            plt.close()
        else:
            shap.image_plot(shap_values=values,
                        pixel_values=data,
                        labels=None)

    def aopc(self, img_index: int, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        SHAP_AOPC = np.array([0.0]*len(percents))

    def get_single_aopc_value(self, img_index: int, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        model = self.predict
        image = self.dataset[img_index]['X'].unsqueeze(0)   # get the base image CxHxW
        if self.opt.loss_type == 'cross_entropy':
            Y_class = self.dataset[img_index]['Y_class']
        else:
            Y_class = np.argmax(self.dataset[img_index]['Y_class']==1)
        Y = self.dataset[img_index]['Y']
        X_pred = model(image)
        base_value = X_pred.softmax(dim=-1).flatten()[Y_class].item()

        AOPC = np.array([0.0]*len(percents))
        for i in range(len(percents)):
            res = self.delete_top_k_features(percents[i], img_index).unsqueeze(0)
            AOPC[i] = base_value - model(res).softmax(dim=-1).flatten()[Y_class]
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/aopc{img_index}_{Y}.npy", AOPC)
        return AOPC
    
    def delete_top_k_features(self, k, img_index: int):
        img = self.dataset[img_index]['X']  # get the image CxHxW
        Y = self.dataset[img_index]['Y']
        path_to_value = f"results/{self.opt.explanation_name}/{self.opt.name}/P{img_index}_{Y}.npy"    # 1 x H x W x C

        value = np.load(path_to_value)
        total_sum = np.sum(value)

        # Sort the values in descending order and get the corresponding indices
        sorted_indices = np.argsort(-value.flatten())
        # sorted_values = value.flatten()[sorted_indices]

        # Calculate the cumulative sum
        # cumulative_sum = np.cumsum(sorted_values)

        # Find the index where the cumulative sum exceeds 50% of the total sum
        # index = np.argmax(cumulative_sum > k * total_sum)

        index = int(k * len(sorted_indices))

        # Set the values after the index to 0
        result = img.flatten()
        result[sorted_indices[:index]] = 0
        result = result.reshape(img.shape)

        return result

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