from options.test_options import TestOptions
from datasets import create_dataloader
from models import create_model
import numpy as np

opt = TestOptions().parse()  # get test options

dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataloader)    # get the number of images in the dataset.
print('The number of testing images = %d' % dataset_size)

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)   

labels = dataloader.dataset.labels   # get the labels so we can search the label from one-hot encoding
for i, data in enumerate(dataloader):
    model.set_input(data)
    model.forward()
    predict_result = model.output.detach().numpy()
    index_max = np.argmax(predict_result)
    print(data['Y'], labels[index_max], data['Y'][0]==labels[index_max], index_max)  # print the true label and the predicted label
'''
python test.py -d ./data/tiny-imagenet -n RestNetClassifier -g -1 -m res_class --dataset_name imagenet --phase val 
'''
