# %% Test ResNet18 on ImageNet
''' (Correctness: 7162/10000)
python test.py -d ./data/tiny-imagenet -n Resnet18onImageNet -g mps -m res_class --dataset_name imagenet --phase val --eval --net_name resnet18 --batch_size 4 --epoch best
'''
# %% Test ResNet50 on ImageNet
''' (Correctness: 7806/10000)
python test.py -d ./data/tiny-imagenet -n Resnet50onImageNet -g mps -m res_class --dataset_name imagenet --phase val --eval --net_name resnet50 --batch_size 4 --epoch best
'''
# %% Test ResNet101 on ImageNet
''' (Correctness: 8602/10000)
python test.py -d ./data/tiny-imagenet -n Resnet101onImageNet -g mps -m res_class --dataset_name imagenet --phase val --eval --net_name resnet101 --batch_size 4 --epoch best
'''
# %% Test ResNet18 on PASCAL_VOC_2012
''' (Correctness: 3227/5823)
python test.py -d ./data/pascal_voc_2012 -n Resnet18onPASCAL -g mps -m res_class --dataset_name pascalvoc2012 --phase val --eval --net_name resnet18 --batch_size 4 --epoch 135 --loss_type bcewithlogits
'''
# %% Test ResNet50 on PASCAL_VOC_2012
''' (Correctness: 4022/5823, 1027/1449)
python test.py -d ./data/pascal_voc_2012 -n Resnet50onPASCAL -g mps -m res_class --dataset_name pascalvoc2012 --phase val --eval --net_name resnet50 --batch_size 4 --epoch best  --loss_type bcewithlogits
'''
# %% Test ResNet101 on PASCAL_VOC_2012
''' (Correctness: 4080/5823, 1044/1449)
python test.py -d ./data/pascal_voc_2012 -n Resnet101onPASCAL -g mps -m res_class --dataset_name pascalvoc2012 --phase val --eval --net_name resnet101 --batch_size 4 --epoch best --loss_type bcewithlogits
'''
# %% Test CNN on Brain-Tumor-MRI
''' (Correctness: 1279/1311, 97.56%)
python test.py --config config/CNNonBrainMRI.yaml --phase val --eval --epoch 20
'''
# %% Test CNN on Icons-50
''' (Correctness: 1564/2000, 78.20%)
python test.py --config config/CNNonIcons50.yaml --phase val --eval --epoch 24
'''

from options.test_options import TestOptions
from options.train_options import TrainOptions
from datasets import create_dataloader
from models import create_model
import time
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

    true_cnt = 0

    labels = dataloader.dataset.labels   # get the labels so we can search the label
    with open(f'{opt.name}_log.txt', 'w') as file:
        for i, data in enumerate(dataloader):
            model.set_input(data)
            time_stamp = time.time() # timer for computation time
            model.test()           # run inference
            # print(f"Computation time: \033[92m{(time.time() - time_stamp)}\033[0m s")
            predict_result = model.output

            if opt.loss_type == 'cross_entropy':
                y_prob = F.softmax(predict_result, dim=-1) # get the probability of each class
            elif opt.loss_type == 'bcewithlogits':
                y_prob = F.sigmoid(predict_result) # get the probability of each class
            else:
                raise NotImplementedError(f'Loss type {opt.loss_type} is not implemented')

            y_pred = torch.where(y_prob > 0.5, torch.tensor(1), torch.tensor(0))

            # index_max = torch.argmax(y_prob, 1) # get the index of the max probability
            # indices = index_max.tolist() # convert the tensor to list
            # predicted_labels = [labels[i] for i in indices] # get the predicted labels

            if opt.loss_type == 'cross_entropy':    # Single label classification
                index_max = torch.argmax(y_prob, 1) # get the index of the max probability
                indices = index_max.tolist() # convert the tensor to list
                predicted_labels = [labels[i] for i in indices] # get the predicted labels
                # is_True = [a == b for a,b in zip(data['Y_class'], index_max)] # check if the predicted label is correct
                is_True = [a == b for a,b in zip(data['label'], index_max)] # check if the predicted label is correct

                for j in range(len(is_True)):
                    # print(f"{i*opt.batch_size+j}\t\033[92m{is_True[j]}\033[0m\t{data['Y'][j]}\t\033[92m{predicted_labels[j]}\033[0m\t{data['Y_class'][j]}\t\033[92m{indices[j]}\033[0m\t{y_prob[j,index_max[j]]}\t{predict_result[j,index_max[j]]}")  # print the true label and the predicted label
                    print(f"{i*opt.batch_size+j}\t\033[92m{is_True[j]}\033[0m\t\033[92m{predicted_labels[j]}\033[0m\t{data['label'][j]}\t\033[92m{indices[j]}\033[0m\t{y_prob[j,index_max[j]]}\t{predict_result[j,index_max[j]]}")  # print the true label and the predicted label

                    # file.write(f"{i*opt.batch_size+j}\t{is_True[j]}\t{data['Y'][j]}\t{predicted_labels[j]}\t{data['Y_class'][j]}\t{indices[j]}\t{y_prob[j,index_max[j]]}\t{predict_result[j,index_max[j]]}\n")
                    file.write(f"{i*opt.batch_size+j}\t{is_True[j]}\t{predicted_labels[j]}\t{data['label'][j]}\t{indices[j]}\t{y_prob[j,index_max[j]]}\t{predict_result[j,index_max[j]]}\n")

                    true_cnt += is_True[j]
            elif opt.loss_type == 'bcewithlogits':   # Multi label classification                
                for j in range(len(y_pred)):
                    predicted_index = torch.where(y_pred[j] == 1)[0]  # get the index of the predicted labels
                    predicted_index = predicted_index.cpu()  # move the tensor to CPU
                    # true_index = torch.where(data['Y_class'][j] == 1)[0] # get the index of the max probability
                    true_index = torch.where(data['label'][j] == 1)[0] # get the index of the max probability

                    predicted_labels = [labels[i] for i in predicted_index.tolist()] # get the predicted labels
                    true_labels = [labels[j] for j in true_index] # get the true labels

                    # is_True = set(predicted_index.tolist()).issubset(set(true_index.tolist())) and len(predicted_index.tolist()) != 0
                    is_True = predicted_index.tolist() == true_index.tolist()
                    true_cnt += is_True
                    # if len(index_predicted) == len(true_index):
                    #     is_True = torch.all(index_predicted == true_index)
                    #     true_cnt += is_True
                    # else:
                    #     is_True = False
                    print(f"{i*opt.batch_size+j}\t\033[92m{is_True}\033[0m\t{predicted_index.tolist()}\t{true_index.tolist()}")
                    
                    file.write(f"{i*opt.batch_size+j}\t{is_True}\t{data['Y'][j]}\t{predicted_index.tolist()}\t{true_index.tolist()}\t{predicted_labels}\t{true_labels}\n")
            else:
                raise NotImplementedError(f'Loss type {opt.loss_type} is not implemented')
        
    
    print(f"\033[92m{true_cnt}\033[0m out of \033[92m{dataset_size}\033[0m are correct")  # print the number of correct predictions
