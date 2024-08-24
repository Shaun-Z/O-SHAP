import numpy as np
import argparse
import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.train_options import TrainOptions
from datasets.imagenet_dataset import ImageNetDataset

parser = argparse.ArgumentParser()
opt = TrainOptions().parse()
dataset = ImageNetDataset(opt)
print(dataset[0]['X'].shape, dataset[0]['Y'])

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(dataset[i]['X'])
    plt.title(dataset[i]['Y'])
    plt.axis('off')
plt.show()

print(len(dataset))