# Introduction

this project is to train a feature extractor by contrasive learning for regression purpose. we adopt contrasive learning method introduced in [a NeurIPS 2023 paper](https://arxiv.org/abs/2210.01189)

# Usage

## Install prerequisites

```shell
python3 -m pip install absl-py torch torchvision tqdm numpy
```

## create dataset

```shell
python3 create_dataset.py --input_dir <path/to/raw/dataset> --output_dir <path/to/processed/dataset>
```

## train on dataset

```shell
python3 train.py --dataset <path/to/processed/dataset> --ckpt <path/to/checkpoint>
```

## plot results

extract features 

```shell
python3 inference.py --input <path/to/raw/trainset/npy> --output trainset.npy --ckpt <path/to/checkpoint> --batch <batch size>
python3 inference.py --input <path/to/raw/evalset/npy> --output evalset.npy --ckpt <path/to/checkpoint> --batch <batch size>
python3 plot.py --trainset trainset.npy --evalset evalset.npy --trainlabel <path/to/raw/trainlabel/npy> --evallabel <path/to/raw/evallabel/npy> --output <path/to/output/png>
```

