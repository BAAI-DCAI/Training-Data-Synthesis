#!/bin/bash
ImageNetPath="path to imagenet"
SyntheticPath="path to synthetic data"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --rdzv_endpoint localhost:5003 train.py \
--imagenet_path $ImageNetPath --data-dir $SyntheticPath \
--dataset imagenette --model resnet50 --num-classes 10 \
--batch-size 128 --opt sgd --weight-decay 5e-4 --sched multistep --lr 0.1 --decay-rate 0.2 --epochs 200 \
--amp --output experiments/synthetic --experiment r50_imagenette --workers 8 --pin-mem --use-multi-epochs-loader 

