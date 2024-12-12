import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from util.color import red_transparent_blue, transparent_red

from models import create_model
from datasets import create_dataset
from .base_explanation import BaseExplanation

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        # np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

class RiseExplanation(BaseExplanation):
    @staticmethod
    def modify_commandline_options(parser):
        # rewrite default values
        # parser.add_argument('--n_evals', type=int, default=5000, help='the number of iterations. The larger the number, the finer the granularity of the significance analysis and the longer the computation consumes time')
        # parser.set_defaults(batch_size=50)
        return parser
    
    def __init__(self, opt):
        super(RiseExplanation, self).__init__(opt)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.dataset = create_dataset(opt)

        self.transform= self.dataset.transform
        self.inv_transform= self.dataset.inv_transform

        self.explainer = self.define_explainer(self.predict, self.dataset)


    def define_explainer(self, pred_fn, dataset):
        Xtr = dataset[0]['X']
        # out = predict(Xtr[0:1])
        
        explainer = RISE(pred_fn, tuple(Xtr[-2,:].shape), gpu_batch=1)
        return explainer
    
    def explain(self, img_index: int):
        self.image = self.dataset[img_index]['X']
        X = self.image.unsqueeze(0)
        indices = self.dataset[img_index]['indices']
        self.class_list = self.dataset[img_index]['indices'] if len(self.opt.index_explain)==0 else self.opt.index_explain
        Y = [self.dataset.labels[i] for i in indices]
        # Class_list = [self.dataset.label2id[l] for l in Y.split(',')]
        indices = self.dataset[img_index]['indices']

        input_img = X
        
        self.explainer.generate_masks(N=500, s=8, p1=0.5)

        self.saliency = self.explainer(input_img.cuda()).cpu().numpy()
        p, c = torch.topk(self.predict(input_img.cuda()), k=2)
        self.p, self.c = p[0], c[0]

        os.makedirs(f"results/{self.opt.explanation_name}/{self.opt.name}/value", exist_ok=True)
        np.save(f"results/{self.opt.explanation_name}/{self.opt.name}/value/P{img_index}_{indices}.npy", self.saliency[self.c[0]])


    def predict(self, img: np.ndarray) -> torch.Tensor:
        self.model.input = nhwc_to_nchw(torch.Tensor(img)).to(self.device)
        self.model.forward()
        # y_prob = F.softmax(self.model.output, dim = -1)
        return F.softmax(self.model.output, dim = -1)
    
    def plot(self, save_path: str = None):
        image = self.inv_transform(self.image).permute(1,2,0)
        image_show = image.mean(axis=-1)

        result_show = self.saliency[self.class_list]
        labels_to_display = [self.dataset.labels[index] for index in self.class_list]

        fig, axes = plt.subplots(nrows=1, ncols=result_show.shape[0]+1, figsize=(8,6), squeeze=False)
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        max_val = np.nanpercentile(np.abs(result_show), 99.9)
        for i in range(result_show.shape[0]):
            axes[0, i+1].set_title(labels_to_display[i])
            axes[0, i+1].imshow(image_show, cmap=plt.get_cmap('gray'))
            # axes[0, i+1].imshow(result_show[i], cmap=transparent_red)
            axes[0, i+1].imshow(result_show[i], cmap='jet', alpha=0.5)
            axes[0, i+1].axis('off')
            # im = axes[0, i+1].imshow(result_show[i], cmap=transparent_red)
            im = axes[0, i+1].imshow(result_show[i], cmap='jet', alpha=0.5)

        cb = plt.colorbar(im, ax=np.ravel(axes).tolist(), label="RISE value", orientation="horizontal", aspect=30)
        cb.outline.set_visible(False)
        

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        


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