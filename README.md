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

