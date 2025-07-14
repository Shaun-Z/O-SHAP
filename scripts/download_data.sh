#!/bin/bash

NAME=$1

echo $NAME

function kaggle_diagnose() {
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
  # pip install kaggle --upgrade
}

case $NAME in
  "carvana")
    kaggle_diagnose
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
  "mnist_csv")
    kaggle_diagnose
    kaggle competitions download -c digit-recognizer
    mkdir -p data/mnist_csv
    unzip digit-recognizer.zip
    mv train.csv data/mnist_csv
    mv test.csv data/mnist_csv
    rm sample_submission.csv
    rm digit-recognizer.zip
    ;;
  "mnist")
    mkdir -p data/mnist/raw
    wget -c -P data/mnist/raw https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    wget -c -P data/mnist/raw https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    wget -c -P data/mnist/raw https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    wget -c -P data/mnist/raw https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    gunzip -k data/mnist/raw/*.gz
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
  "cub200")
    kaggle_diagnose
    kaggle datasets download veeralakrishna/200-bird-species-with-11788-images --unzip
    mkdir -p data/cub200
    tar -xvzf CUB_200_2011.tgz -C ./data/cub200
    tar -xvzf segmentations.tgz -C ./data/cub200
    rm CUB_200_2011.tgz
    rm segmentations.tgz
    ;;
  "severstal")
    kaggle_diagnose
    kaggle competitions download -c severstal-steel-defect-detection
    mkdir -p data/severstal
    tar -xvzf severstal-steel-defect-detection.zip -C ./data/severstal
    rm severstal-steel-defect-detection.zip
    ;;
  "pascal_voc_2007")
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    mkdir -p data/pascal_voc_2007/trainval
    mkdir -p data/pascal_voc_2007/test
    tar -xvf VOCtrainval_06-Nov-2007.tar -C ./data/pascal_voc_2007/trainval
    tar -xvf VOCtest_06-Nov-2007.tar -C ./data/pascal_voc_2007/test
    rm VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar
    ;;
  "pascal_voc_2012")
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    # wget -c http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
    mkdir -p data/pascal_voc_2012/trainval
    # mkdir -p data/pascal_voc_2012/test
    tar -xvf VOCtrainval_11-May-2012.tar -C ./data/pascal_voc_2012/trainval
    # tar -xvf VOC2012test.tar -C ./data/pascal_voc_2012/test
    rm VOCtrainval_11-May-2012.tar
    # rm VOCtrainval_11-May-2012.tar VOC2012test.tar
    ;;
  "icons50")
    curl -L -o Icons50.zip https://www.kaggle.com/api/v1/datasets/download/danhendrycks/icons50
    mkdir -p data/icons50
    unzip -o Icons50.zip -d data/icons50
    rm Icons50.zip
    ;;
  "brain_tumor_mri")
    curl -L -o brain-tumor-mri.zip https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset
    mkdir -p data/brain_tumor_mri
    unzip brain-tumor-mri.zip -d data/brain_tumor_mri
    rm brain-tumor-mri.zip
    ;;
  "sgcc")
    git clone https://github.com/henryRDlab/ElectricityTheftDetection.git
    mkdir -p data/sgcc
    7z x ./ElectricityTheftDetection/data.zip -o./data/sgcc
    rm -rf ./ElectricityTheftDetection
    ;;
  "eurosat")
    wget -c http://madm.dfki.de/files/sentinel/EuroSAT.zip
    mkdir -p data/eurosat
    unzip EuroSAT.zip -d data/eurosat
    rm EuroSAT.zip
    ;;
  "jatracer")
    curl -L -o jatracer-images.zip https://www.kaggle.com/api/v1/datasets/download/conorsully1/jatracer-images
    mkdir -p data/jatracer
    unzip jatracer-images.zip -d data/jatracer
    rm jatracer-images.zip
    ;;
  "malaria")
    curl -L -o malaria-bounding-boxes.zip https://www.kaggle.com/api/v1/datasets/download/kmader/malaria-bounding-boxes
    unzip malaria-bounding-boxes.zip -d data
    rm malaria-bounding-boxes.zip
    rm -r ./data/malaria/malaria
    ;;
  "mvtec_ad")
    curl -L -o mvtec_ad.zip https://www.kaggle.com/api/v1/datasets/download/ipythonx/mvtec-ad
    mkdir -p data/mvtec_ad
    unzip mvtec_ad.zip -d data/mvtec_ad
    rm mvtec_ad.zip
    ;;
  "lisa_traffic_light")
    curl -L -o ./lisa-traffic-light-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mbornoe/lisa-traffic-light-dataset
    mkdir -p data/lisa_traffic_light
    unzip ./lisa-traffic-light-dataset.zip -d data/lisa_traffic_light
    rm ./lisa-traffic-light-dataset.zip
    ;;
  "celeba")
    curl -L -o ~/Downloads/celeba-dataset.zip  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
    mkdir -p data/celeba
    unzip ~/Downloads/celeba-dataset.zip -d data/celeba
    rm ~/Downloads/celeba-dataset.zip
    ;;
  "adult")
    curl -L -o ./adult-census-income.zip https://www.kaggle.com/api/v1/datasets/download/uciml/adult-census-income
    mkdir -p data/adult_census_income
    unzip ./adult-census-income.zip -d data/adult_census_income
    rm ./adult-census-income.zip
    ;;
  *)
    echo "Invalid dataset name"
    ;;
esac
