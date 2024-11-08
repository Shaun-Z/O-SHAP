from tqdm import tqdm
from .base_model import BaseModel
from .unet.unet_parts import *
from util.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from . import networks

class UNet1DModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay (L2 penalty)')
        parser.add_argument('--momentum', type=float, default=0.999, help='momentum term of RMSprop')
        parser.add_argument('--amp', action='store_true', help='use mixed precision training')
        parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay (L2 penalty)')
        parser.add_argument('--momentum', type=float, default=0.999, help='momentum term of RMSprop')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        return parser
    
    def __init__(self, opt):
        super(UNet1DModel, self).__init__(opt)
        self.loss_names = ['UNet']
        self.visual_names = ['data', 'true_masks']
        self.model_names = ['UNet']
        
        self.n_classes = 2
        self.amp = opt.amp
        self.netUNet = networks.define_unet_masker(opt.input_nc, opt.num_classes, False, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()

            # initialize optimizers; 
            # self.optimizer = torch.optim.Adam(self.netUNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.RMSprop(self.netUNet.parameters(),
                              lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum, foreach=True)
            self.optimizers.append(self.optimizer)

            # Function <BaseModel.setup> has been overwritten by <UNet1DModel.setup>
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
            # Disable AMP if using MPS
            self.grad_scaler = torch.amp.GradScaler(device=self.device, enabled=self.amp)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        
        self.data = input['data'].to(self.device)
        self.true_masks = input['mask'].to(self.device)

    def train(self):
        """Make models train mode during test time"""
        for param in self.netUNet.parameters():
            param.requires_grad = True

    
    def validate(self, DataLoader_val):
        # Set model to evaluation mode
        self.eval()
        num_val_batches = len(DataLoader_val)
        dice_score = 0

        # iterate over the validation set
        with torch.autocast(self.device.type, enabled=self.amp):
            for i, data in enumerate(DataLoader_val):
                
                # data, mask_true = batch['data'].to(self.device), batch['mask'].to(self.device)
                # # predict the mask
                # mask_pred = self.netUNet(data)

                self.set_input(data)
                self.forward()
                mask_pred = self.masks_pred
                mask_true = self.true_masks

                if self.netUNet.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < self.netUNet.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, self.netUNet.n_classes).permute(0, 2, 1).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.netUNet.n_classes).permute(0, 2, 1).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
        val_score = dice_score / max(num_val_batches, 1)
        self.metric = val_score
        # Set model back to training mode
        self.train()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.masks_pred = self.netUNet(self.data)  # model(data)

    #
    def backward(self):
        """Calculate the loss for back propagation"""
        # First, G_A should fake the discriminator
        self.loss_UNet = self.criterion(self.masks_pred, self.true_masks)
        self.loss_UNet += dice_loss(
                    F.softmax(self.masks_pred, dim=1).float(),
                    F.one_hot(self.true_masks, self.netUNet.n_classes).permute(0, 2, 1).float(),
                    multiclass=True)
        self.grad_scaler.scale(self.loss_UNet).backward()
        
    # Overwrite <BaseModel.optimize_parameters> function
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions
        self.set_requires_grad([self.netUNet], True)
        # backward
        self.optimizer.zero_grad()  # set gradients to zero
        self.backward()             # calculate gradients
        self.grad_scaler.unscale_(self.optimizer)
        self.grad_scaler.step(self.optimizer)       
        self.grad_scaler.update()   # update weights

    # Overwrite <BaseModel.setup> function
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            print("Setting model to \033[92mtrain\033[0m mode")
            self.schedulers = [self.scheduler]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    # Overwrite <BaseModel.update_learning_rate> function
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
            # scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

