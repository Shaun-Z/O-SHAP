'''
python explain.py -d ./data/tiny-imagenet -n CustomClassifier -g 0 -m res_class --net_name custom --dataset_name imagenet --eval --explanation_name shap --epoch 15

python explain.py -d ./data/tiny-imagenet -n Resnet50onImageNet -g mps -m res_class --net_name resnet50 --dataset_name imagenet --eval --explanation_name bhem --epoch 15

python explain.py -d ./data/tiny-imagenet -n Resnet18onImageNet -g mps -m res_class --net_name resnet18 --dataset_name imagenet --eval --explanation_name shap --epoch 25

python explain.py -d ./data/pascal_voc_2007 -n Resnet18onPASCAL -g mps -m res_class --net_name resnet18 --dataset_name pascalvoc --eval --explanation_name shap --epoch 35 --num_classes 20 --loss_type bcewithlogits
'''

import time
from tqdm import tqdm
from options.explain_options import ExplainOptions
from explanations import create_explanation

if __name__ == '__main__':
    opt = ExplainOptions().parse()   # get explain options
    explainer = create_explanation(opt)
    # img_index = 5
    for img_index in tqdm(range(50)):
        # explainer.explain(img_index) # 1, 5
        explainer.get_single_aopc_value(img_index)
        Y_class = explainer.dataset[img_index]['Y_class']
        Y = explainer.dataset[img_index]['Y']
        print(Y_class, Y)
        explainer.plot(0, save_path=f"results/{opt.explanation_name}/{opt.name}/P{img_index}_{Y}.svg")
    