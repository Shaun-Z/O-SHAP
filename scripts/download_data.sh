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
  "imagenet")
    kaggle datasets download -d puneet6060/intel-image-classification
    mkdir -p data/imagenet
    unzip intel-image-classification.zip
    mv seg_train data/imagenet
    mv seg_test data/imagenet
    mv seg_pred data/imagenet
    rm -r seg_train
    rm -r seg_test
    rm -r seg_pred
    rm intel-image-classification.zip
    ;;
  *)
    echo "Invalid dataset name"
    ;;
esac
