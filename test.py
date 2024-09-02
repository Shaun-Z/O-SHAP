from options.test_options import TestOptions
from options.train_options import TrainOptions
from datasets import create_dataloader
from models import create_model
import numpy as np
import torch
import torch.nn.functional as F

opt = TestOptions().parse()  # get test options
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)
dataset_size = len(dataloader)    # get the number of images in the dataset.
print('The number of testing images = %d' % dataset_size)

# test with eval mode. This only affects layers like batchnorm and dropout.
if opt.eval:
    model.eval()

labels = dataloader.dataset.labels   # get the labels so we can search the label from one-hot encoding
for i, data in enumerate(dataloader):
    model.set_input(data)
    model.test()
    predict_result = model.output
    y_prob = F.softmax(predict_result, dim = -1)
    index_max = torch.argmax(y_prob, 1) #np.argmax(predict_result)
    is_True = data['Y'][0]==labels[index_max]
    print(f"\033[92m{is_True}\033[0m\t{data['Y']}\t\033[92m{labels[index_max]}\033[0m\t{data['Y_class']}\t\033[92m{index_max}\033[0m\t{y_prob[0,index_max]}")  # print the true label and the predicted label
    # exit()
'''
python test.py -d ./data/tiny-imagenet -n RestNetClassifier -g -1 -m res_class --dataset_name imagenet --phase val --eval --net_name resnet101
'''
