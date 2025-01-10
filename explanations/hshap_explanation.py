import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import hshap

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation
from util.color import red_transparent_blue
from util.util import nhwc_to_nchw, nchw_to_nhwc

class HShapExplanation(BaseExplanation):

    @staticmethod
    def modify_commandline_options(parser):
        # rewrite default values
        parser.add_argument('--ss', type=int, default=57, help='the number of iterations. The larger the number, the finer the granularity of the significance analysis and the longer the computation consumes time')
        parser.add_argument('--threshold_mode', type=str, default='absolute', help='thresholding modes')
        parser.add_argument('--threshold', type=float, default=0.0, help='thresholding values')
        return parser
    
    def __init__(self, opt):
        super(HShapExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)

        self.transform= torchvision.transforms.Compose([
            transforms.Lambda(nchw_to_nhwc),
        ])
        self.inv_transform= torchvision.transforms.Compose([
            transforms.Lambda(nhwc_to_nchw),
            transforms.Normalize(
                mean = (-1 * np.array(self.dataset.mean) / np.array(self.dataset.std)).tolist(),
                std = (1 / np.array(self.dataset.std)).tolist()
            ),
            transforms.Lambda(nchw_to_nhwc),
        ])

        self.explainer = self.define_explainer(self.model, self.dataset)


    def explain(self, img_index: int):
        self.image = self.dataset[img_index]['X']
        X = self.image.unsqueeze(0)
        label = self.dataset[img_index]['label']
        indices = self.dataset[img_index]['indices']
        self.class_list = self.dataset[img_index]['indices']

        input_img = X.squeeze().to(self.device)   # torch.Size([1, 3, 224, 224])
        
        output_indexes = self.class_list

        # explain image
        self.explanation = self.explainer.explain(
            input_img,
            label=label,
            s=self.opt.ss,
            threshold_mode=self.opt.threshold_mode,
            threshold=self.opt.threshold,
            batch_size=1,
        )

        os.makedirs(f"results/{self.opt.explanation_name}/{self.opt.name}/value", exist_ok=True)
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/value/P{img_index}_{indices}.npy", self.explanation.squeeze().cpu().detach().numpy())

    def define_explainer(self, pred_fn, dataset):
        Xtr = dataset[0]['X']
        ref = torch.zeros_like(Xtr).to(self.device)
        hexp = hshap.Explainer(pred_fn, ref)
        return hexp
    
    # def predict(self, img: np.ndarray) -> torch.Tensor:
    #     self.model.input = nhwc_to_nchw(torch.Tensor(img)).to(self.device)
    #     self.model.forward()
    #     y_prob = F.softmax(self.model.output, dim = -1)
    #     return y_prob

    def plot(self, save_path: str = None):
        image = self.inv_transform(self.image)

        image_show = image.mean(axis=-1)
        result_show = self.explanation
        # print(image_show.shape, result_show.shape)
        # exit()
        labels_to_display = [self.dataset.labels[index] for index in self.class_list]

        fig, axes = plt.subplots(nrows=1, ncols=result_show.shape[0]+1, figsize=(8,6), squeeze=False)
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        max_val = np.nanpercentile(np.abs(result_show), 99.9)
        for i in range(result_show.shape[0]):
            axes[0, i+1].set_title(labels_to_display[i])
            axes[0, i+1].imshow(image_show, cmap=plt.get_cmap('gray'))
            axes[0, i+1].imshow(result_show[i], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)
            # axes[0, i+1].imshow(result_show[i], cmap='jet', alpha=0.5)
            axes[0, i+1].axis('off')
            im = axes[0, i+1].imshow(result_show[i], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)

        cb = plt.colorbar(im, ax=np.ravel(axes).tolist(), label="Gradient SHAP value", orientation="horizontal", aspect=30)
        cb.outline.set_visible(False)
        

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()