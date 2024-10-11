# %% Explain Resnet18 on ImageNet using SHAP
'''
python explain.py -d ./data/tiny-imagenet -n CustomClassifier -g 0 -m res_class --net_name custom --dataset_name imagenet --eval --explanation_name shap --epoch 15
'''
# %% Explain Resnet50 on ImageNet using BHEM
'''
python explain.py -d ./data/tiny-imagenet -n Resnet50onImageNet -g mps -m res_class --net_name resnet50 --dataset_name imagenet --eval --explanation_name bhem --epoch 15
'''
# %% Explain Resnet18 on ImageNet using SHAP
'''
python explain.py -d ./data/tiny-imagenet -n Resnet18onImageNet -g mps -m res_class --net_name resnet18 --dataset_name imagenet --eval --explanation_name shap --epoch 25
'''
# %% Explain Resnet50 on PASCAL_VOC_2012 using BHEM
'''
python explain.py -d ./data/pascal_voc_2012 -n Resnet50onPASCAL -g mps -m res_class --net_name resnet50 --dataset_name pascalvoc2012 --eval --explanation_name bhem --epoch best --loss_type bcewithlogits --segmentation --approx
'''

import time
from tqdm import tqdm
from options.explain_options import ExplainOptions
from explanations import create_explanation, aopc

if __name__ == '__main__':
    opt = ExplainOptions().parse()   # get explain options

    # set indexes for explanation
    opt.index_explain = [int(i) for i in opt.index_explain]

    explainer = create_explanation(opt)
    # img_index = 1

    # time_stamp = time.time()
    # explainer.explain(img_index) # 1, 5
    # print(f"Computation time: \033[92m{(time.time() - time_stamp)}\033[0m s")
    # # aopc.get_single_aopc_value(explainer.predict, explainer.dataset, img_index, opt.explanation_name, opt.name)

    # Y_class = explainer.dataset[img_index]['Y_class']
    # Y = explainer.dataset[img_index]['Y']
    # print(Y_class, Y)

    # # explainer.plot(save_path=f"results/{opt.explanation_name}/{opt.name}/image/P{img_index}_{Y}.png")
    # explainer.plot()


    for img_index in tqdm(range(200)):
        Y_class = explainer.dataset[img_index]['Y_class']
        Y = explainer.dataset[img_index]['Y']
        # print(Y_class, Y)
        explainer.explain(img_index)
        explainer.plot(save_path=f"results/{opt.explanation_name}/{opt.name}/image/P{img_index}_{Y}.png")
        
        # aopc.get_single_aopc_value(explainer.predict, explainer.dataset, img_index, opt.explanation_name, opt.name)
        
    
    # aopc.get_average_aopc_value(opt.explanation_name, opt.name)