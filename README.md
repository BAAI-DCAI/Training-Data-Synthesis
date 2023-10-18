## Real-Fake: Effective Training Data Synthesis Through Distribution Matching [PDF](https://arxiv.org/pdf/2310.10402.pdf)

Synthetic training data has gained prominence in numerous learning tasks and scenarios, offering advantages such as dataset augmentation, generalization evaluation, and privacy preservation. Despite these benefits, the efficiency of synthetic data generated by current methodologies remains inferior when training advanced deep models exclusively, limiting its practical utility. To address this challenge, we analyze the principles underlying training data synthesis for supervised learning and elucidate a principled theoretical framework from the distribution-matching perspective that explicates the mechanisms governing synthesis efficacy. Through extensive experiments, we demonstrate the effectiveness of our synthetic data across diverse image classification tasks, both as a replacement for and augmentation to real datasets, while also benefits challenging tasks such as out-of-distribution generalization and privacy preservation.


## Installation

The project has been tested with PyTorch 2.01 and CUDA 11.7.

### Install Required Environment

```bash
pip3 install -r requirements.txt
```

## Prepare Dataset

Download ImageNet-1K from [this link](https://www.image-net.org/download.php).

## (Optional) Download Generated Synthetic Dataset

We will shortly release the synthetic data.

## Extract CLIP Embedding for ImageNet-1K

1. Check `./extract.sh` and specify the path to the ImageNet data.

```bash
bash extract.sh
```

## Get BLIP2 Caption for ImageNet-1K

Use the implementation of the BLIP2 caption pipeline. Refer to [this paper](https://arxiv.org/abs/2307.08526) for details.

## Train LoRA

1. Specify `CACHE_ROOT/MODEL_NAME` to the folder caching stable diffusion.
2. Check `./LoRA/train_lora.sh` and specify the data version in "versions" for training LoRA.

```bash
bash ./LoRA/train_lora.sh
```

## Generate Synthetic Dataset

1. After training, load the trained LoRA model to generate the Synthetic Dataset.
2. Check `shell_generate.sh` and specify the data version (1 out of 20) in "versions" for generation.
3. Review the parameter `--nchunks 8` (Number of GPUs, for example, 8).

```bash
bash shell_generate.sh
```

This will save one version of the dataset to `./SyntheticData`.

## Evaluate

1. Check `train.sh` and specify `--data_dir` with "version" for training on the generated synthetic data.
2. Review `CUDA_VISIBLE_DEVICES=0,1,2` and `--nproc_per_node=3` to specify the number of GPUs used.

```bash
bash train.sh
```

This will save results and the model to `./experiments/`.
