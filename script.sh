#!/bin/bash

# export MODEL_PATH=/nfs/scistore19/alistgrp/ecrncevi/SpQR/llama/llama-2-7b/consolidated.00.pth
export MODEL_PATH=/media/elvircrn/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/SpQR/Llama-2-7b-hf
export DATASET=/media/elvircrn/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/SpQR/AQLM/data/red_pajama_n=4096_4096_context_length_llama.pth
# export DATASET=/nfs/scistore19/alistgrp/ecrncevi/SpQR/data/red_pajama_n=1024.pth

python3 main.py $MODEL_PATH $DATASET \
    --wbits 3 \
    --groupsize 16 \
    --perchannel \
    --qq_scale_bits 3 \
    --qq_zero_bits 3 \
    --qq_groupsize 16 \
    --outlier_threshold=0.2 \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 \
    --save /media/elvircrn/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/SpQR/output0
