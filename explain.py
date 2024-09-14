'''
python explain.py -d ./data/tiny-imagenet -n CustomClassifier -g 0 -m res_class --net_name custom --dataset_name imagenet --eval --explanation_name shap --epoch 15

python explain.py -d ./data/tiny-imagenet -n Resnet50onImageNet -g mps -m res_class --net_name resnet50 --dataset_name imagenet --eval --explanation_name bhem --epoch 15
'''

import time
from tqdm import tqdm
from options.explain_options import ExplainOptions
from explanations import create_explanation

if __name__ == '__main__':
    opt = ExplainOptions().parse()   # get explain options
    explainer = create_explanation(opt)
    explainer.explain(5) # 1, 5
    print(explainer.dataset[5]['Y_class'], explainer.dataset[5]['Y'])
    explainer.plot()
    