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
'''
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.imagenet_dataset import ImageNetDataset

from datasets import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = create_dataset(opt)

    # Load the data item with index 200
    index = 100
    data = dataset[index]  # Load the 200th sample

    # Extract normalized image and box_map
    normalized_image = data['X']  # Tensor for the normalized image
    box_map = data['mask']  # Tensor for the box map

    # Perform inverse normalization (recover the original image)
    inv_image = dataset.inv_transform(normalized_image)  # Reverse the normalization
    image_pil = ToPILImage()(inv_image)  # Convert to PIL image

    # Convert box_map tensor to PIL image
    box_map_pil = ToPILImage()(box_map)

    if image_pil.mode != 'RGBA':
        image_pil = image_pil.convert('RGBA')
    if box_map_pil.mode != 'RGBA':
        box_map_pil = box_map_pil.convert('RGBA')
    # Overlay box_map on the original image
    overlay = Image.alpha_composite(image_pil, box_map_pil)

    # Display the combined image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(f"Label: {data['label']}, Indices: {data['indices']}")
    plt.axis("off")
    plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.celeba_dataset import CelebADataset  # Ensure this is the correct path to the CelebA dataset

from datasets import create_dataset

if __name__ == '__main__':
    # Parse the test options
    opt = TestOptions().parse()

    # Create the dataset instance using the options provided
    dataset = create_dataset(opt)

    # Print the total number of samples in the dataset
    print(f"Dataset size: {len(dataset)}")

    # Loop through the first 5 samples of the dataset for demonstration
    for i in range(5):  # Adjust this range if you want to process more samples
        data = dataset[i]
        print(f"Sample {i + 1}:")
        print(f"  X shape: {data['X'].shape}")  # Shape of the image tensor
        print(f"  Label tensor: {data['label']}")  # Binary label tensor
        print(f"  Indices of labels: {data['indices']}\n")  # Indices of active labels (1s)

        # Visualize the image with its corresponding active label indices
        img = data['X'].permute(1, 2, 0).numpy()  # Convert the image tensor to a NumPy array
        img = (img * 0.5 + 0.5)  # Denormalize the image back to [0, 1] range
        plt.imshow(img)
        plt.title(f"Labels: {data['indices']}")  # Display active label indices as the title
        plt.axis('off')  # Remove axes for better visualization
        plt.show()

    # Print all possible labels in the dataset
    # Assumes the dataset has an attribute `labels` containing label names
    print("Dataset labels:")
    print(dataset.labels)

