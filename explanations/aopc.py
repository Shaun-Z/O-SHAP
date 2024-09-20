import numpy as np
import os

def get_single_aopc_value(model, dataset, img_index, explanation_name, name, percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        # model = self.predict
        image = dataset[img_index]['X']   # get the base image 1xCxHxW
        Y_class = dataset[img_index]['Y_class']
        Y = dataset[img_index]['Y']
        Class_list = dataset[img_index]['Class_list']
        
        X_pred = model(image.unsqueeze(0))
        base_value = X_pred.softmax(dim=-1).flatten()[Y_class].item()

        AOPC = np.array([0.0]*len(percents))
        for i in range(len(percents)):
            res = delete_top_k_features(dataset, img_index, explanation_name, name, percents[i]).unsqueeze(0)
            AOPC[i] = base_value - model(res).softmax(dim=-1).flatten()[Y_class]
        os.makedirs(f"results/{explanation_name}/{name}/AOPC", exist_ok=True)
        np.save(f"results/{explanation_name}/{name}/AOPC/aopc{img_index}_{Y}.npy", AOPC)
        return AOPC
    
def delete_top_k_features(dataset, img_index: int, explanation_name, name, percent):
    img = dataset[img_index]['X']  # get the image CxHxW
    Y = dataset[img_index]['Y']
    path_to_value = f"results/{explanation_name}/{name}/value/P{img_index}_{Y}.npy"    # 1 x H x W x C

    value = np.load(path_to_value)

    # Sort the values in descending order and get the corresponding indices
    sorted_indices = np.argsort(-value.flatten())

    index = int(percent * len(sorted_indices))

    # Set the values after the index to 0
    result = img.flatten()
    result[sorted_indices[:index]] = 0
    result = result.reshape(img.shape)

    return result