#!/bin/bash

dataset='imagenette'
versions=('v1')
loras=('gt_dm')
methods=('SDI2I_LoRA')
guidance_tokens=('Yes')
SDXLs=('No')
image_strengths=(0.75)

length=${#versions[@]}
echo "start Generation Loop"
for ((i=0; i<$length; i++)); do
    ver="${versions[$i]}"
    lora="./LoRA/checkpoint/${loras[$i]}_${ver}"
    method="${methods[$i]}"
    guidance_token="${guidance_tokens[$i]}"
    SDXL="${SDXLs[$i]}"
    cw="${cws[$i]}"
    imst="${image_strengths[$i]}"
    echo "$ver LoRA: $lora Method $method"
    # Iterate from 0-7, cover all case for nchunks <= 8
    for j in {0..7}; do
        echo $j
        CUDA_VISIBLE_DEVICES=$j python generate.py --index ${j} --method $method --version $ver --batch_size 24 \
        --use_caption "blip2" --dataset $dataset --lora_path $lora --if_SDXL $SDXL --use_guidance $guidance_token \
        --img_size 512 --cross_attention_scale 0.5 --image_strength $imst --nchunks 8 > results/gen${j}.out 2>&1 &
    done
    wait
    echo "All processes completed"
done