#!/bin/bash

export MODEL_PATH=
export DATASET=

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
    --save <output_path>
