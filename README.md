# Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction

## Overview

The Dataset and code for paper [Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction](https://arxiv.org/abs/2312.12021), accepted by AAAI 2024.

![1703063577738](images/1703063577738.png)

## Requirements

```
GPU=NVIDIA A100 Tensor Core
Python=3.7
Pytorch=1.13.0
```

## Pre-training

```shell
cd pre-train/code
bash train.sh
```

## Fine-tuning

Step1. Select the down-stream baseline and train

```shell
cd fine-tune
bash run_train.sh
```

Step2. Select the down-stream baseline and test

```shell
cd fine-tune
bash run_test.sh
```

## Citation

```
@inproceedings{luo2024synergistic,
  title={Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction},
  author={Luo, Da and Gan, Yanglei and Hou, Rui and Lin, Run and Liu, Qiao and Cai, Yuxiang and Gao, Wannian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={18742--18750},
  year={2024}
}
```

