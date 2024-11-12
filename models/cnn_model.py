import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks

class CnnModel(BaseModel):
    """A simple CNN model for classification task.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_classes', type=int, default=4, help='the number of output image classes')
        parser.add_argument('--loss_type', type=str, default='cross_entropy', help='the type of loss function [cross_entropy | bcewithlogits]')
        parser.set_defaults(lr=0.001)
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_CNN_val = 0

        self.loss_names = ['CNN', 'CNN_val']
        self.visual_names = ['input']

        if self.isTrain:
            self.model_names = ['Cnn_classifier']
        else:
            self.model_names = ['Cnn_classifier']

        self.netCnn_classifier = networks.define_cnn_classifier(opt.input_nc, opt.num_classes, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            if opt.loss_type == 'cross_entropy':
                self.criterion = torch.nn.CrossEntropyLoss()    # Softmax + NLLLoss
            elif opt.loss_type == 'bcewithlogits':
                self.criterion = torch.nn.BCEWithLogitsLoss()   # Sigmoid + BCELoss
            else:
                raise NotImplementedError(f'Loss type {opt.loss_type} is not implemented')
            
        self.optimizer = torch.optim.Adam(self.netCnn_classifier.parameters(), lr=opt.lr)
        self.optimizers.append(self.optimizer)
        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.input = input['X'].to(self.device)
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output = self.netCnn_classifier(self.input)

    def validate(self, DataLoader_val):
        loss = 0
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(DataLoader_val):
                self.set_input(data)
                self.forward()
                loss += self.criterion(self.output, self.label)
        self.loss_CNN_val = loss / i
        self.train()

    def backward(self):
        """Calculate the loss for back propagation"""
        # First, G_A should fake the discriminator
        self.loss_CNN = self.criterion(self.output, self.label)
        self.loss_CNN.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions
        self.set_requires_grad([self.netCnn_classifier], True)
        # backward
        self.optimizer.zero_grad()  # set gradients to zero
        self.backward()             # calculate gradients
        self.optimizer.step()       # update weights

# class CnnClassifier(nn.Module):
#     def __init__(self, num_classes, dropout=0.5):
#         super(CnnClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d(p=dropout)
#         self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
#         self.fc2 = nn.Linear(100, 10)
#         self.fc1_drop = nn.Dropout(p=dropout)
#         # self.cnn = cnn(dropout)
#         # self.fc = nn.Linear(10, num_classes)

#     def forward(self, x):
#         x = torch.relu(F.max_pool2d(self.conv1(x), 2))
#         x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

#         # flatten over channel, height and width = 1600
#         x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

#         x = torch.relu(self.fc1_drop(self.fc1(x)))
#         x = torch.softmax(self.fc2(x), dim=-1)
#         return x