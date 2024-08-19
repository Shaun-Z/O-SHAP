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
  "other")
    # Add code for other dataset here
    ;;
  *)
    echo "Invalid dataset name"
    ;;
esac

# kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
# unzip train_hq.zip
# mv train_hq/* data/imgs/
# rm -d train_hq
# rm train_hq.zip

# kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
# unzip train_masks.zip
# mv train_masks/* data/masks/
# rm -d train_masks
# rm train_masks.zip
