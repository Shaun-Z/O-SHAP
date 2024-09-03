from options.test_options import TestOptions
from options.train_options import TrainOptions
from datasets import create_dataloader
from models import create_model
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    print(f'The number of testing images = \033[92m{dataset_size}\033[0m')

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    labels = dataloader.dataset.labels   # get the labels so we can search the label
    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        predict_result = model.output

        y_prob = F.softmax(predict_result, dim = -1) # get the probability of each class
        index_max = torch.argmax(y_prob, 1) # get the index of the max probability
        indices = index_max.tolist() # convert the tensor to list
        predicted_labels = [labels[i] for i in indices] # get the predicted labels

        is_True = [a == b for a,b in zip(data['Y'], predicted_labels)] # check if the predicted label is correct

        for j in range(len(is_True)):
            print(f"\033[92m{is_True[j]}\033[0m\t{data['Y'][j]}\t\033[92m{predicted_labels[j]}\033[0m\t{data['Y_class'][j]}\t\033[92m{indices[j]}\033[0m\t{y_prob[j,index_max[j]]}")  # print the true label and the predicted label
'''
python test.py -d ./data/tiny-imagenet -n ResnetClassifier -g -1 -m res_class --dataset_name imagenet --phase val --eval --net_name resnet101 --batch_size 4
'''
