from options.test_options import TestOptions
from datasets import create_dataset
from models import create_model
import numpy as np

opt = TestOptions().parse()  # get test options

dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
print('The number of testing images = %d' % dataset_size)

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)   

for i, data in enumerate(dataset):
    print(data['X'].shape, data['Y'])
    model.set_input(data)
    model.forward()
    print(np.argmax(model.output.detach().numpy()))
    exit()

'''
python test.py -d ./data/tiny-imagenet -n RestNetClassifier -g -1 -m res_class --dataset_name imagenet --phase val 
'''