# ML-Testbench

This is a testbench for ML algorithms and explanation methods.

> These have already been achieved
> - Dataset
>   - [ ] MNIST
>   - [X] Tiny-ImageNet
>   - [X] Brain-Tumor-MRI
>   - [ ] CUB-200-2011
> - Model
>   - [X] CNN Classifier
>   - [X] Resnet Classifier

- [ML-Testbench](#ml-testbench)
- [Configuration YAML Files](#configuration-yaml-files)
- [Download Datasets](#download-datasets)
- [Models](#models)
  - [ResNet Classifier](#resnet-classifier)
  - [CNN Classifier](#cnn-classifier)
- [Training](#training)
  - [ImageNet](#imagenet)
  - [Brain Tumor MRI](#brain-tumor-mri)
- [How to add your work](#how-to-add-your-work)
  - [Model](#model)
  - [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

# Configuration YAML Files
This is a newly added function and has not been fully tested.

The configuration files are stored as `./config/<config>.yaml`. They provide arguments for each python commands you run

```bash
python <filename>.py --config <path_to_yaml>
```

After specifying `--config`, you can also add other arguments, e.g.

```bash
python <filename>.py --config <path_to_yaml> --eval
```

> The priority of arguments in YAML files are higher than that of manually specified arguments. In other words, if you specify the argument `dataroot` in both **YAML** and **command line**, the final value of `dataroot` will be the value in YAML.

# Download Datasets

1. carvana-image-masking-challenge

```bash
bash scripts/download_data.sh carvana
```

2. MNIST

```bash
bash scripts/download_data.sh mnist
```

3. ImageNet

```bash
bash scripts/download_data.sh tiny-imagenet
```

4. cifar10

```bash
bash scripts/download_data.sh cifar10
```

5. cub200

```bash
bash scripts/download_data.sh cub200
```

6. serverstal

```bash
bash scripts/download_data.sh serverstal
```

7. pascal_voc_2007

```bash
bash scripts/download_data.sh pascal_voc_2007
```

# Models

## ResNet Classifier

`./models/res_class_model.py`

## CNN Classifier

`./models/cnn_model.py`

# Training

## ImageNet

```bash
python train.py --dataroot ./data/tiny-imagenet --name Restnet101Classifier --gpu_ids -1 --model res_class --net_name resnet101 --dataset_name imagenet --batch_size 128
```

or

```bash
python train.py -d ./data/tiny-imagenet -n Restnet101Classifier -g -1 -m res_class --net_name resnet101 --dataset_name imagenet --batch_size 128
```

> Some additional options
> - `--continue_train`: load the latest model
> - `--load_iter`: specify which iteration to load
> - `--save_by_iter`: save model by iteration
> - `--use_wandb`: use wandb for visualization

## Brain Tumor MRI
```bash
python train.py --config ./config/CNNonBrainMRI.yaml
```

# How to add your work

## Model

1. Put your model file under `./models/`

    Remember to name your file as `{model_name}_model.py` to make it looks like:

    ```bash
    ./models/{model_name}_model.py
    ```

2. Write **basic network structures** in `./models/networks.py`

    For example, `Resnet` and `Unet`:

    ```python
    class ResnetBlock(nn.Module):
        """Define a Resnet block"""

        def __init__(self, **args):

            super(ResnetBlock, self).__init__()
            self.conv_block = self.build_conv_block(**args)

        def build_conv_block(self, **args):
            pass

        def forward(self, x):
            pass
    ```

3. (optional) Write functions in `networks.py` to define how to get parts of your model
    ```python
    def define_G(**args):
        net = ResnetGenerator(**args)
        return init_net(net, **args)
    ```

4. Create your model

    Import those basic networks in your model file `{model_name}_model.py`
    ```python
    from . import networks
    ```

    Define your class (Example: `ResClassModel`)
    ```python
    class ResClassModel(BaseModel):
        '''Please add options in this method if you want more customized options for your model while traing or testing.'''
        @staticmethod
        def modify_commandline_options(parser, is_train=True):
            parser.set_defaults(no_dropout=True)  # default model did not use dropout

            parser.add_argument('--{option_name}', type={data_type}, default={default_value}, help='{help message}')
            
            return parser
        
        def __init__(self, opt):
            """Initialize the ResNet classificaion class.

            Parameters:
                opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            """
            BaseModel.__init__(self, opt)
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['Resnet']    # Strings in this list should correspond to names of losses of this class. You can find it in `def backward():` part. 

            self.visual_names = ['input']  # Strings in this list should correspond to names of any property of this class.

            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
            if self.isTrain:
                self.model_names = ['Resnet_classifier'] # Strings in this list should correspond to names of the networks that you want to use in training phase.
            else:  # during test time, only load xxx
                self.model_names = ['Resnet_classifier'] # Strings in this list should correspond to names of the networks that you want to use in validation/testing phase.

            # define networks
            self.netResnet_classifier = networks.define_resnet_classifier(opt.input_nc, opt.num_classes, opt.ngf, opt.net_name, opt.norm, not opt.no_dropout, opt.pool_type, opt.init_type, opt.init_gain, self.gpu_ids) # The name of this property should be in the form of `net{self.model_names}`

            if self.isTrain:
                # define loss functions
                self.criterion = torch.nn.CrossEntropyLoss()  # define loss.
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer = torch.optim.Adam(self.netResnet_classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer) # List of optimizers

        def set_input(self, input):
            """Unpack input data from the dataloader and perform necessary pre-processing steps.

            Parameters:
                input (dict): include the data itself and its metadata information.
            """
            self.input = input['X'].to(self.device)
            self.label = input['Y_class'].to(self.device)
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']

        def forward(self):
            """Run forward pass; called by both functions <optimize_parameters> and <test>."""
            self.output = self.netResnet_classifier(self.input)  # model(input)

        def backward(self):
            """Calculate the loss for back propagation"""
            # First, G_A should fake the discriminator
            self.loss_Resnet = self.criterion(self.output, self.label)
            self.loss_Resnet.backward()

        def optimize_parameters(self):
            """Calculate losses, gradients, and update network weights; called in every training iteration"""
            # forward
            self.forward()      # compute predictions
            self.set_requires_grad([self.netResnet_classifier], True)
            # backward
            self.optimizer.zero_grad()  # set gradients to zero
            self.backward()             # calculate gradients
            self.optimizer.step()       # update weights
    ```

## Dataset

TBD

# Acknowledgements
I would like to express my sincere gratitude to the following individuals and projects for their contributions and inspiration to this project:

- [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6#2.1-Load-data): The MNIST dataset preprocessing for this project references its methodology.

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): The structure of this repository references this project.

I am truly thankful for the open-source community and the invaluable resources they provide.

- [pytorch-image-classification](https://github.com/bentrevett/pytorch-image-classification): Some methods from this repository are used.