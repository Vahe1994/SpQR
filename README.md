# SPQR model compression


**Note:** This repository contains quantization algorithm and the model evaluation code for SpQR method for LLM compression; 
The efficient inference code will be added soon.
    
It accompanies the research paper "[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)" .

# Installation

### Packages

To run SpQR with `falcon` make sure that you have `torch>=2.0.0` with `CUDA` support.

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

__Note:__ the results reported in the ArXiv paper where obtained using `4.28.dev0` version of `transformers`, commit id [`464d420775`](https://github.com/huggingface/transformers/archive/464d420775653885760e30d24d3703e14f4e8a14.zip).


### Loading / caching datasets and tokenizer

The script will require downloading and caching locally the relevant tokenizer and the datasets. They will be saved in `DATASETS_CACHE` directory.

### Models

This repository is expected to work with one of the `LLaMA` or `Falcon` models.

#### Data

For quantization with SpQR its is recommended to use the subset of the data model 
was trained on. I.e for quantization of `LLaMA` models we recommend to use the subset
of [RedPajamas](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) and for `Falcon` quantization - (RefinedWeb)[https://huggingface.co/datasets/tiiuae/falcon-refinedweb]. Both subsets 
are stored in `data` directory: 
* `data/red_pajama_n=1024.pth`
* `data/refined_web_n=128.pth`
  
**Note** These subsets are already processed with the corresponding model tokenizer. Use for different model will lead to
unexpected behavior.

### W&B logging

For the sake of convenience one can optionally log the data to `wandb`.
Run `pip install wandb` for W&B logging.
Specify `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_NAME` environmental prior to running experiments.

# Launching

### GPU requirements
This code was developed and tested using a single A100 GPU with 80GB GPU RAM. It may successfully run on GPUs with 32 - 40GB   

### Model downloading
The code requires the LLaMA model to be dowloaded in Hugging Face format and saved locally. The scripts below require such model folder path as argument.

### Perplexity benchmarks:
This script compresses the model and then tests its performance in terms of perplexity using WikiText2, 
C4, and Penn Treebank datasets. 

The command to launch the script should look like this: 

```
export MODEL_PATH=<PATH_TO_MODEL_DIR>

python main.py $MODEL_PATH custom \
    --custom_data_path=PATH_TO_CUSTOM_DATA \
    --wbits 4 \
    --groupsize 16 \
    --perchannel \
    --qq_scale_bits 3 \
    --qq_zero_bits 3 \
    --qq_groupsize 16 \
    --outlier_threshold=0.2 \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 
```
The command above runs near-lossless compression as described in the article. Adjusting the above parameters allows for tighter compression with a slightly greater loss. 

Note the launch arguments:
- `<PATH_TO_MODEL_DIR>` - path to model folder, which contains `config.json `
- `one of [c4, ptb, wikitext2, custom]` -- name of dataset to use for compression
- `--custom_data_path` - path to preprocessed and tokenized dataset (if `custom` chosen). Otherwise do not specify.
- `--wbits 3` -- number of bits for quantized weights representation
- `--groupsize 16` -- size of first-order groups for compression
- `--qq_groupsize 16` -- size of second-order (quantized) groups for compression
- `--qq_scale_bits 3 --qq_zero_bits 3` -- bit sizes for quantizing first order weights' scale and zeros.
run `python main.py --help` for more details on command line arguments, including compression parameters.

### LM Evaluation Harness benchmark.

To perform zero-shot evaluation, we use [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework with slight modifications. 

For instructions about zero-shot evaluation refer to `README.md` inside `lm-evaluation-harness` directory.

## Citation
```
@misc{dettmers2023spqr,
      title={SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression}, 
      author={Tim Dettmers and Ruslan Svirschevski and Vage Egiazarian and Denis Kuznedelev and Elias Frantar and Saleh Ashkboos and Alexander Borzunov and Torsten Hoefler and Dan Alistarh},
      year={2023},
      eprint={2306.03078},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```