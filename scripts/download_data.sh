#!/bin/bash

NAME=$1

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade

echo $NAME

case $NAME in
  "carvana")
    kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
    mkdir -p data/carvana/imgs
    unzip train_hq.zip
    mv train_hq/* data/carvana/imgs/
    rm -r train_hq
    rm train_hq.zip

    kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
    mkdir -p data/carvana/masks
    unzip train_masks.zip
    mv train_masks/* data/carvana/masks/
    rm -r train_masks
    rm train_masks.zip
    ;;
  "mnist")
    kaggle competitions download -c digit-recognizer
    mkdir -p data/mnist
    unzip digit-recognizer.zip
    mv train.csv data/mnist
    mv test.csv data/mnist
    rm sample_submission.csv
    rm digit-recognizer.zip
    ;;
  "cifar10")
    wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    mkdir -p data/cifar10
    tar -xvf cifar-10-python.tar.gz
    mv cifar-10-batches-py/* data/cifar10
    rm cifar-10-python.tar.gz
    rm -r cifar-10-batches-py
    ;;
  "tiny-imagenet")
    wget -c https://www.image-net.org/data/tiny-imagenet-200.zip
    mkdir -p data/tiny-imagenet
    unzip tiny-imagenet-200.zip
    mv tiny-imagenet-200/* data/tiny-imagenet
    rm tiny-imagenet-200.zip
    rm -r tiny-imagenet-200
    ;;
  *)
    echo "Invalid dataset name"
    ;;
esac
