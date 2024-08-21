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

# Test Scripts
1. MNIST dataset test
```bash
python mnist_dataset_test.py --dataroot ./data/mnist
```

# Models
## cnn_model

# Acknowledgements
I would like to express my sincere gratitude to the following individuals and projects for their contributions and inspiration to this project:

- [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6#2.1-Load-data): The MNIST dataset preprocessing for this project references its methodology.

- [CycleGAN](https://github.com/junyanz/CycleGAN): The structure of this repository references this project.

I am truly thankful for the open-source community and the invaluable resources they provide.
