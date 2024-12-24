'''
To test this script, run the following command:
----------------
python imagenet_dataset_test.py --dataroot ./data/tiny-imagenet --gpu_ids -1
----------------
or
----------------
python imagenet_dataset_test.py -d ./data/tiny-imagenet -g -1
----------------
'''
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.imagenet_dataset import ImageNetDataset

from datasets import create_dataset

if __name__ == '__main__':
    
    opt = TestOptions().parse()
    dataset = create_dataset(opt)

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"X:{data['X'].shape}\tlabel:{data['label']}\tindices:{data['indices']}")

    print(dataset.labels)
'''
'''
To test this script, run the following command:
----------------
python imagenet_dataset_test.py --dataroot ./data/tiny-imagenet --gpu_ids -1
----------------
or
----------------
python imagenet_dataset_test.py -d ./data/tiny-imagenet -g -1
----------------
'''
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image

from options.train_options import TrainOptions
from options.test_options import TestOptions

from datasets import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = create_dataset(opt)

    # Load the data item with index 200
    index = 0
    data = dataset[index]  # Load the 200th sample

    # Extract normalized image and box_map
    normalized_image = data['X']  # Tensor for the normalized image
    # mask = data['mask']  # Tensor for the box map
    bounding = data['bounding']  # Tensor for the bounding box

    # Perform inverse normalization (recover the original image)
    inv_image = dataset.inv_transform(normalized_image)  # Reverse the normalization
    image_pil = ToPILImage()(inv_image)  # Convert to PIL image

    # Convert box_map tensor to PIL image
    bounding_pil = ToPILImage()(bounding)  # Convert to PIL image

    if image_pil.mode != 'RGBA':
        image_pil = image_pil.convert('RGBA')
    if bounding_pil.mode != 'RGBA':
        bounding_pil = bounding_pil.convert('RGBA')
    # Overlay box_map on the original image
    overlay = Image.alpha_composite(image_pil, bounding_pil)

    # Display the combined image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(f"Label: {data['label']}, Indices: {data['indices']}")
    plt.axis("off")
    plt.show()
