# ML-Testbench

This is a testbench for ML algorithms and explanation methods.

# Download Datasets

1. carvana-image-masking-challenge

```bash
bash scripts/download_data.sh carvana
```

2. MNIST

```bash
bash scripts/download_data.sh mnist
```

3. ImageNet

```bash
bash scripts/download_data.sh imagenet
```
# Test Scripts

1. MNIST

```bash
python mnist_dataset_test.py --dataroot ./data/mnist --gpu_ids -1
```

or

```bash
python mnist_dataset_test.py -d ./datasets/mnist -g -1
```

2. ImageNet

```bash
python imagenet_dataset_test.py --dataroot ./data/tiny-imagenet --gpu_ids -1
```

or

```bash
python imagenet_dataset_test.py -d ./data/tiny-imagenet -g -1
```

# Models

## ResNet Classifier

`./models/res_class_model.py`

# Training

## ImageNet

```bash
python train.py --dataroot ./data/tiny-imagenet --name RestNetClassifier --gpu_ids -1 --model res_class --dataset_name imagenet
```

or

```bash
python train.py -d ./data/tiny-imagenet -n RestNetClassifier -g -1 -m res_class --dataset_name imagenet
```

> Some additional options
> - `--continue_train`: load the latest model
> - `--load_iter`: specify which iteration to load
> - `--save_by_iter`: save model by iteration

# Acknowledgements
I would like to express my sincere gratitude to the following individuals and projects for their contributions and inspiration to this project:

- [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6#2.1-Load-data): The MNIST dataset preprocessing for this project references its methodology.

- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): The structure of this repository references this project.

I am truly thankful for the open-source community and the invaluable resources they provide.
