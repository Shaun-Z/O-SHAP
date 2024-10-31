from .base_model import BaseModel
from .unet.unet_parts import *
from util.dice_score import dice_loss

class UNet1DModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay (L2 penalty)')
        parser.add_argument('--momentum', type=float, default=0.999, help='momentum term of RMSprop')
        parser.set_defaults(lr=1e-4)  # default learning rate
        return parser
    
    def __init__(self, opt):
        super(UNet1DModel, self).__init__(opt)
        self.loss_names = ['UNet']
        self.visual_names = ['data', 'mask']
        self.model_names = ['UNet']
        
        n_classes = 2
        self.netUNet = UNet_1D(n_channels=1, n_classes=n_classes, bilinear=False)

        if self.isTrain:
            # define loss functions
            self.criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()

            # initialize optimizers; 
            # self.optimizer = torch.optim.Adam(self.netUNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.RMSprop(self.netUNet.parameters(),
                              lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum, foreach=True)
            self.optimizers.append(self.optimizer)

            # TODO: schedulers will be overwritten by function <BaseModel.setup>.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
            self.grad_scaler = torch.amp.GradScaler(device=self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        
        self.data = input['data'].to(self.device)
        self.true_masks = input['mask'].to(self.device)

    # TODO: Implement the train function
    def train(self):
        """Make models train mode during test time"""
        for param in self.netUNet.parameters():
            param.requires_grad = False
        for param in self.netUNet.layer4.parameters():
            param.requires_grad = True
        self.netUNet.fc.requires_grad = True

    # TODO: Implement the validate function
    def validate(self, DataLoader_val):
        loss = 0
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(DataLoader_val):
                self.set_input(data)
                self.forward()
                loss += self.criterion(self.output, self.label)
        self.loss_Resnet_val = loss / i
        self.train()

    # TODO: Implement the forward function
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.masks_pred = self.netUNet(self.data)  # model(data)

    # TODO: Implement the backward function
    def backward(self):
        """Calculate the loss for back propagation"""
        # First, G_A should fake the discriminator
        self.loss_UNet = self.criterion(self.masks_pred, self.true_masks)
        self.loss_UNet += dice_loss(
                    F.softmax(self.masks_pred, dim=1).float(),
                    F.one_hot(self.true_masks, self.netUNet.n_classes).permute(0, 2, 1).float(),
                    multiclass=True)
        self.grad_scaler.scale(self.loss_Resnet).backward()
        
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

    # TODO: Overwrite <BaseModel.update_learning_rate> function
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step(self.metric)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

class UNet_1D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv_1D(n_channels, 32))
        self.down1 = (Down_1D(32, 64))
        self.down2 = (Down_1D(64, 128))
        factor = 2 if bilinear else 1
        self.down3 = (Down_1D(128, 256 // factor))
        self.up1 = (Up_1D(256, 128 // 1, bilinear))
        self.up2 = (Up_1D(128, 64 // 1, bilinear))
        self.up3 = (Up_1D(64, 32 // 1, bilinear))
        self.outc = (OutConv_1D(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
