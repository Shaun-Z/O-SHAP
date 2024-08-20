import numpy as np
import argparse
import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.train_options import TrainOptions
from datasets.mnist_dataset import MnistDataset
parser = argparse.ArgumentParser()
opt = TrainOptions().parse()
dataset = MnistDataset(opt)
print(dataset[0]['X_train'].shape, dataset[0]['Y_train'], dataset[0]['X_val'].shape, dataset[0]['Y_val'])

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(dataset[i]['X_train'], cmap='gray')
    plt.title(np.argmax(dataset[i]['Y_train']))
    plt.axis('off')
plt.show()