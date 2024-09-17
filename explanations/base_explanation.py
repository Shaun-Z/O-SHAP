import os
import torch
from abc import ABC, abstractmethod
from pathlib import Path

class BaseExplanation():
    
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('mps' if self.gpu_ids == 'mps' else 'cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        Path(f"results/{opt.explanation_name}/{self.opt.name}").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def modify_commandline_options(parser):
        # parser.add_argument('--phase', type=str, default='test', help='during explanation, the phase is always test')
        return parser
    
    @abstractmethod
    def explain(self, data, model):
        pass

    @abstractmethod
    def get_single_aopc_value(self, img_index: int, percents: list):
        pass

    @abstractmethod
    def save(self, path):
        pass
