'''
python explain.py -d ./data/tiny-imagenet -n CustomClassifier -g 0 -m res_class --net_name custom --dataset_name imagenet --eval
--epoch 15
'''

import time
from tqdm import tqdm
from options.explain_options import ExplainOptions
from explanations import create_explanation

if __name__ == '__main__':
    opt = ExplainOptions().parse()   # get explain options
    explainer = create_explanation(opt)
    explainer.explain(1)
    explainer.plot()