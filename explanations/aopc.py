import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

# %% get the average AOPC value (It requires the AOPC values to be saved in the results folder)
def get_average_aopc_value(explanation_name, name, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    path_to_aopc = f"results/{explanation_name}/{name}/AOPC"
    aopc_files = os.listdir(path_to_aopc)
    aopcs = np.stack([np.load(os.path.join(path_to_aopc, file)) for file in aopc_files], axis=0)
    average_aopc = np.mean(aopcs, axis=0)
    os.makedirs(f"results/{explanation_name}/{name}", exist_ok=True)
    np.save(f"results/{explanation_name}/{name}/average_aopc.npy", average_aopc)
    plt.plot(percents, average_aopc)
    plt.xlabel("Percentage of deleted features")
    plt.ylabel("AOPC")
    plt.title("Average AOPC")
    plt.savefig(f"results/{explanation_name}/{name}/average_aopc.png")
    plt.close()
    return average_aopc

def get_single_aopc_value(model, dataset, img_index, explanation_name, name, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        # model = self.predict
        image = dataset[img_index]['X']   # get the base image 1xCxHxW
        Y_class = dataset[img_index]['Y_class']
        Y = dataset[img_index]['Y']
        Class_list = dataset[img_index]['get_class_list'](Y_class)
        
        X_pred = model(image.unsqueeze(0))
        base_value = X_pred.softmax(dim=-1).flatten()[Y_class].item()

        AOPC = np.array([0.0]*len(percents))
        for i in range(len(percents)):
            res = delete_top_k_features(dataset, img_index, explanation_name, name, percents[i]).unsqueeze(0)
            os.makedirs(f"results/{explanation_name}/{name}/delete_image", exist_ok=True)
            plt.figure()
            plt.imshow(dataset.inv_transform(res[0]).permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.savefig(f"results/{explanation_name}/{name}/delete_image/P{img_index}_{Y}_{percents[i]}.png")
            plt.close()
            AOPC[i] = base_value - model(res).softmax(dim=-1).flatten()[Y_class]
        os.makedirs(f"results/{explanation_name}/{name}/AOPC", exist_ok=True)
        np.save(f"results/{explanation_name}/{name}/AOPC/aopc{img_index}_{Y}.npy", AOPC)
        return AOPC
    
def delete_top_k_features(dataset, img_index: int, explanation_name, name, percent):
    img = dataset[img_index]['X']  # get the image CxHxW
    Y = dataset[img_index]['Y']
    path_to_value = f"results/{explanation_name}/{name}/value/P{img_index}_{Y}.npy"    # 1 x H x W x C

    value = np.load(path_to_value).mean(axis=-1)  # H x W
    
    # Sort the values in descending order and get the corresponding indices
    sorted_indices = np.argsort(-value.flatten())

    index = int(percent * len(sorted_indices))

    # Set the values after the index to 0
    result = img.clone()
    result = result.reshape(img.shape[0], -1)

    for i in range(img.shape[0]):
        result[i,...] = img[i,...].flatten()
        result[i,...][sorted_indices[:index]] = 0
    
    result = result.reshape(img.shape)

    return result