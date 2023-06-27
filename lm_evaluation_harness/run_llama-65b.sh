#!/bin/bash
export TRANSFORMERS_CACHE=/extra_disk_1/yozh/LLaMA-65B/transformers_cache
export CUDA_VISIBLE_DEVICES=1,3,5
export OMP_NUM_THREADS=16

export WANDB_ENTITY=rock-and-roll
export WANDB_PROJECT=LLaMA-Compression
export WANDB_NAME=llama-65b_3.518_bits

NSAMPLES=128

python lm_eval_main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-65b-hf,dtype=float16,cache_dir=/extra_disk_1/yozh/LLaMA-65B/transformers_cache,use_accelerate=True,max_memory_per_gpu=48GIB \
    --quantization_args dataset=custom,custom_data_path=data/red_pajama_n=1024.pth,wbits=3,groupsize=16,perchannel=True,qq_scale_bits=3,qq_zero_bits=3,qq_groupsize=64,percdamp=1.0,outlier_threshold=0.45,nsamples=128,offload_activations=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1 \
    --no_cache \
    --log_wandb 