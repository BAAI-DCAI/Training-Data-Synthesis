#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python extract_feature.py --index 0 --imagenet_path "PATH TO ImageNet-1K"

wait
echo "All processes completed"