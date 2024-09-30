# %% Evaluate BHEM on Resnet18 on PASCAL_VOC_2007
'''
python evaluate_xai.py -d ./data/pascal_voc_2007 -n Resnet18onPASCAL -g mps -m res_class --net_name resnet18 --dataset_name pascalvoc --eval --explanation_name bhem --epoch 75 --num_classes 20 --loss_type bcewithlogits --segmentation --num_test 200
'''
# %% Evaluate BHEM on Resnet50 on PASCAL_VOC_2007
'''
python evaluate_xai.py -d ./data/pascal_voc_2007 -n Resnet50onPASCAL -g mps -m res_class --net_name resnet50 --dataset_name pascalvoc --eval --explanation_name bhem --epoch 80 --num_classes 20 --loss_type bcewithlogits --segmentation --num_test 200
'''

'''
python evaluate_xai.py -d ./data/pascal_voc_2007 -n Resnet101onPASCAL -g mps -m res_class --net_name resnet101 --dataset_name pascalvoc --eval --explanation_name shap --epoch 75 --num_classes 20 --loss_type bcewithlogits --segmentation --num_test 200
'''

import numpy as np
import glob
import os
from tqdm import tqdm
from pathlib import Path
from options.explain_options import ExplainOptions
from explanations import create_explanation, metrics

if __name__ == '__main__':
    opt = ExplainOptions().parse()   # get explain options
    save_path = Path(f'results/metric_xai/{opt.explanation_name}/{opt.name}')
    (save_path/"EBPG").mkdir(parents=True, exist_ok=True)
    (save_path/"mIoU").mkdir(parents=True, exist_ok=True)
    (save_path/"Bbox").mkdir(parents=True, exist_ok=True)
    # set indexes for explanation
    # opt.index_explain = [int(i) for i in opt.index_explain]

    explainer = create_explanation(opt)
    dataset = explainer.dataset
    labels_all = dataset.labels
    print(labels_all)

    for img_ID in tqdm(range(opt.num_test)):
        Y_class = dataset[img_ID]["Y_class"]
        Y = dataset[img_ID]["Y"]
        X = dataset[img_ID]["X"]
        mask = dataset[img_ID]["mask"]
        
        mask_int = np.array((mask*255)).astype(np.uint8)
        mask_int.shape, np.unique(mask_int)

        label_index = np.unique(mask_int)[1:-1]-1
        label = [labels_all[i] for i in label_index]

        file_path = next(Path(f'results/{opt.explanation_name}/{opt.name}/value/').glob(f'P{img_ID}_*.npy'))    # May be changed in the future
        xai_result = np.load(file_path)

        if isinstance(label_index, int):
            ebpg = metrics.calculate_ebpg(xai_result, mask_int.squeeze(),label_index+1)
            miou = metrics.calculate_miou(xai_result, mask_int.squeeze(), label_index+1)
            bbox = metrics.calculate_bbox(xai_result, mask_int.squeeze(), label_index+1)
        else:
            ebpg = [metrics.calculate_ebpg(xai_result, mask_int.squeeze(),label_index[i]+1) for i in range(len(label_index))]
            miou = [metrics.calculate_miou(xai_result, mask_int.squeeze(), label_index[i]+1) for i in range(len(label_index))]
            bbox = [metrics.calculate_bbox(xai_result, mask_int.squeeze(), label_index[i]+1) for i in range(len(label_index))]
    
        np.save(save_path/'EBPG'/f'P{img_ID}_{Y}.npy', np.array(ebpg))
        np.save(save_path/'mIoU'/f'P{img_ID}_{Y}.npy', np.array(miou))
        np.save(save_path/'Bbox'/f'P{img_ID}_{Y}.npy', np.array(bbox))

    if os.path.exists(f'results/metric_xai/bhem/{opt.name}') and os.path.exists(f'results/metric_xai/shap/{opt.name}'):
        for metric_name in ["EBPG", "mIoU", "Bbox"]:
            metric_bhem_list = glob.glob(f'results/metric_xai/bhem/{opt.name}/{metric_name}/*.npy')
            metric_shap_list = glob.glob(f'results/metric_xai/shap/{opt.name}/{metric_name}/*.npy')

            metric_bhem = 0
            metric_shap = 0
            for i in range(len(metric_bhem_list)):
                metric_bhem += np.load(metric_bhem_list[i]).mean()
                metric_shap += np.load(metric_shap_list[i]).mean()
            
            print(f"\033[92m{metric_name}\033[0m\nBHEM: {metric_bhem/len(metric_bhem_list)}\nSHAP: {metric_shap/len(metric_shap_list)}")
        
    
        