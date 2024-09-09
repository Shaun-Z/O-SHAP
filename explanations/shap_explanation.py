import numpy as np
import torch
import torchvision
from torchvision import transforms
import shap

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation

class ShapExplanation(BaseExplanation):
    
    @staticmethod
    def modify_commandline_options(parser):
        # default values
        parser.add_argument('--n_evals', type=int, default=5000, help='the number of iterations. The larger the number, the finer the granularity of the significance analysis and the longer the computation consumes time')
        parser.add_argument('--index_explain', type=list, default=[100], help='the shape of the input data')

        # rewrite default values
        parser.set_defaults(batch_size=50)
        return parser

    def __init__(self, opt):
        super(ShapExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)
        self.explainer = self.define_explainer(opt, self.dataset)

    def explain(self, X: np.ndarray):
        input_img = transform(X)
        self.shap_values = self.explainer(input_img, max_evals=self.opt.n_evals, batch_size=self.opt.batch_size, outputs=self.opt.index_explain)

    def predict(self, img: np.ndarray) -> torch.Tensor:
        self.model.input = torch.Tensor(img).to(self.device)
        self.model.forward()
        return self.model.output

    def define_explainer(self, dataset):
        Xtr = transform(dataset[0]['X'].unsqueeze(0))
        # out = predict(Xtr[0:1])
        masker_blur = shap.maskers.Image("blur(64, 64)", Xtr.shape)

        explainer = shap.Explainer(self.predict, masker_blur, output_names=None)
        return explainer
    
    def save(self, path):
        pass

    def plot(self):
        shap.image_plot(shap_values=self.shap_values.values,
                        pixel_values=self.shap_values.data,
                        labels=None)

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
    transforms.Lambda(nchw_to_nhwc),
    transforms.Resize(224),
    transforms.Lambda(lambda x: x*(1/255)),
    transforms.Normalize(mean=mean, std=std),
]

inv_transform= [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Normalize(
        mean = (-1 * np.array(mean) / np.array(std)).tolist(),
        std = (1 / np.array(std)).tolist()
    ),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)