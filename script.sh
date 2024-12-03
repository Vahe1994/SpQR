#!/bin/bash

export MODEL_PATH=
export DATASET=

PYTHONPATH=. python3 main.py /home/steinbms/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 /home/steinbms/SpQR/data/red_pajama_n=1024.pth
    --wbits 3
    --groupsize 16
    --perchannel
    --qq_scale_bits 3
    --qq_zero_bits 3
    --qq_groupsize 16
    --outlier_threshold=0.2
    --permutation_order identity
    --percdamp 1e0
    --nsamples 128
    --save /home/steinbms/matze_experiments/elvircrncevic/spqr/data/sparsity/model0