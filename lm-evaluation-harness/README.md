# Language Model Evaluation Harness

## Overview

The code, utilities and assets located in this directory are adapted from [LM Evaluation Harness benchmark suite](https://github.com/EleutherAI/lm-evaluation-harness) and customized to support quantization. The LICENSE and CODEOWNERS files inside lm-evaluation-harness refer to the original authors of lm-eval-harness and not the anonymous authors of this paper.

The workflow involves following steps:
- Model quantization
- Running tasks from the benchmarks for the quantized model

## Installation

Before running the code make sure that you have all the requirements and dependencies of `lm-eval-harness` installed.
To install them run `
```bash
pip install -r requirements.txt
```

## Data preparation

For calibration we use the same data as in the root directory.

## Execution

The main script lauching the evaluation procedure is `lm_eval_main.py`.

**Note**. Current version of the script support only LLaMA quantization. Therefore set:
* `--model=hf-causal`
* `--model_args pretrained=$MODEL_PATH` where `$MODEL_PATH` has to be one of the LLaMA models
  
`--quantization_args` - list of comma separated arguments for quantizer. For details and options
refer to `lm_eval/quantization/config.py`.  

Below is presented an example of benchmark launch.

```
export MODEL_PATH=<INSERT PATH_TO_MODEL_DIR>
export PAJAMAS_PATH=<INSERT PATH TO PAJAMAS DIR>

python lm_eval_main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --quantization_args dataset=custom,custom_data_path=$PAJAMAS_PATH,wbits=4,groupsize=16,perchannel=True,qq_scale_bits=3,qq_zero_bits=3,qq_groupsize=16,percdamp=1.0,outlier_threshold=0.2,simplified_outliers=False,nsamples=128,offload_activations=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1
```

Performance and runtime notes:
* For large models (LLaMA-30B, LLaMA-65B) specify `max_memory_per_gpu={value}GIB` so that there are free 15-20GIB of GPU memory for each GPU to store activations for calibration. 
* `offload_activations=True` slightly reduces peak memory consumption 
* Typically `30B` requires 1-2 A100 GPUs with 80Gb of memory and `65B` 3 A100.
    

## Citation

BibTeX citation of the original lm-eval-harness repository.

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
