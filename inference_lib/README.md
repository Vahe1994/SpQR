# SpQR model compression


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

The script will require downloading and caching locally the relevant tokenizer and the datasets. 
They will be saved in default Huggingface Datasets directory unless alternative location is provided by env variables.
See [relevant Datasets documentation section](https://huggingface.co/docs/datasets/main/en/cache#cache-directory)
### Models

This repository is expected to work with models of `LLaMA`, `Falcon` and `OPT` families so far.

#### Data

For quantization with SpQR its is recommended to use the subset of the data model 
was trained on. I.e. for quantization of `LLaMA` models we recommend to use the subset
of [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) and for `Falcon` quantization - [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb).Both subsets  are stored in `data` directory: 
* `data/red_pajama_n=1024.pth`
* `data/refined_web_n=128.pth`
  
**Note** These subsets are already processed with the corresponding model tokenizer. Use for different model will lead to
unexpected behavior.

 For `OPT` following GPTQ paper we recommend to use `c4`. 

### W&B logging

For the sake of convenience one can optionally log the data to `Weights and Biases` service (wandb).
Run `pip install wandb` for W&B logging.
Specify `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables prior to running experiments. use `--wandb` argument to enable logging

# Launching

### GPU and RAM requirements
This code was developed and tested using a single A100 GPU with 80GB GPU RAM. It may successfully run on GPUs with 32GB+ VRAM for perplexity evaluation of up to `LLaMA-65B` and `Falcon-40B` models. 
With `--offload activations` option, the model perplexity may be evaluated on machines with less VRAM: 24GB+ for Llama 65B and 6GB+ for Llama 7B.
The perplexity testing code also requires RAM amount sufficient to hold uncompressed model weights (e.g. ~130GB for Llama65B) and testing datasets.
For `Language Model Evaluation Harness` evaluation one needs to have enough memory to load whole model
on one or several devices + activation tensors.

### Model downloading
The code requires the LLaMA model to be downloaded in Huggingface format and saved locally. The scripts below assume that `$TRANSFORMERS_CACHE` variable points to the Huggingface Transformers cache folder.

### Perplexity benchmarks:
This script compresses the model and then tests its performance in terms of perplexity using WikiText2, C4, and Penn Treebank datasets. 

The command to launch the script should look like this: 

```
export MODEL_PATH=<PATH_TO_MODEL_DIR>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>

python main.py $MODEL_PATH $DATASET \
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
- `one of [c4, ptb, wikitext2, pajama, refinedweb, none]` -- name of dataset to use for compression, or path to an alternative preprocessed and tokenized dataset.
- `--wbits 3` -- number of bits for quantized weights representation
- `--groupsize 16` -- size of first-order groups for compression
- `--qq_groupsize 16` -- size of second-order (quantized) groups for compression
- `--qq_scale_bits 3 --qq_zero_bits 3` -- bit sizes for quantizing first order weights' scale and zeros.
- `--offload activations` -- moves activations to RAM when not used. Reduces VRAM usage while slowing work by ~10%. 
run `python main.py --help` for more details on command line arguments, including compression parameters.
- `--save --load` -- path to save/load quantized model.
### LM Evaluation Harness benchmark.

To perform zero-shot evaluation, we use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework with slight modifications. This repository contains a copy of LM Evaluation Harness repo from early 2023 in `lm-eval-harness` folder. 
#### Installation
Before running the code make sure that you have all the requirements and dependencies of `lm-eval-harness` installed. To install them run:
```
pip install -r lm-evaluation-harness/requirements.txt
```
#### Execution

The main script launching the evaluation procedure is `lmeval.py` .

Note. Current version of the script support only LLaMA/Falcon quantization. Therefore, set:
* `--model=hf-causal`
* `--model_args pretrained=$MODEL_PATH` where `$MODEL_PATH` has to be one of the LLaMA models
  
`--quantization_args` - list of comma separated arguments for quantizer. For details and options
refer to `spqr_config.py`.

Below is presented an example of benchmark launch.

```
export MODEL_PATH=<INSERT PATH_TO_MODEL_DIR>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>

python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --quantization_args dataset=$DATASET,wbits=4,groupsize=16,perchannel=True,qq_scale_bits=3,qq_zero_bits=3,qq_groupsize=16,percdamp=1.0,outlier_threshold=0.2,simplified_outliers=False,nsamples=128,offload_activations=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1
```

Performance and runtime notes:
* For large models (LLaMA-30B, LLaMA-65B) specify `max_memory_per_gpu={value}GIB` so that there are free 15-20GIB of GPU memory for each GPU to store activations for calibration. 
* `offload_activations=True` slightly reduces peak memory consumption 
* Typically `LlaMA-30B` requires 1-2 A100 GPUs with 80Gb of memory and `LlaMA-65B` requires 3 A100 with 80Gb each.
* With enough spare GPU memory, one can raise batch size to accelerate evaluation process.


## Inference

This repository also contains an efficient CUDA kernel implementation of the 
SpQR matvec. The file `inference_demo.py` h orcontains a demo of this functionality 
by running end-to-end model inference. Below is an example of how to launch it.

```bash
usage: inference_demo.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH] [--compressed_model_path COMPRESSED_MODEL_PATH] --execution_mode {0,1}

options:
  -h, --help            show this help message and exit
  --pretrained_model_path PRETRAINED_MODEL_PATH
                        Path to the model to the pretrained model
  --compressed_model_path COMPRESSED_MODEL_PATH
                        Path to the compressed .pt model
  --execution_mode {0,1}
                        If set to 0, will evaluate the dense pretrained model. If set to 1, will evaluate the spqr-quantized model
```

This script also reports the mean and median time of the forward() passes and the total inference execution time. 

# Pre-Requisites for Running the Conversion Scripts, Tests and Benchmarks

In order to run the benchmark and test suite you need to build the sources used by these scripts.
You can do so by running the following command:

```bash
/bin/bash scripts/build.sh 
```

which simply runs the `setup.py` script.

# Conversion From Legacy to Optimized SPQR Storage

After running SpQR which produces the tensors stored in int8, in order to run the efficient inference kernels, 
one must convert the tensors produces by SpQR (legacy tensors) into the optimized storage format used by 
the cuda kernel. In order to do so, run the following script:

```bash
usage: convert_legacy_model_format.py [-h] --base_model BASE_MODEL --legacy_model_path LEGACY_MODEL_PATH [--sparse_strategy {csr,ptcsr,optimize_latency}] [--save_pt SAVE_PT] [--save_per_layer SAVE_PER_LAYER]

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        path or name of the unquantized model
  --legacy_model_path LEGACY_MODEL_PATH
                        path to legacy model
  --sparse_strategy {csr,ptcsr,optimize_latency}
                        Sparse strategy storage. Options: csr, ptcsr, auto. CSR - Compressed Sparse Rows PTCSR - Alternative storage format optimize_latency - Use the current GPU to determine the optimal storage format to reduce
                        kernel latency
  --save_pt SAVE_PT     Save the converted quantized .pt model here
  --save_per_layer SAVE_PER_LAYER
                        Save the converted quantized m
```

# Hugginface Conversion

To convert a model into a Hugging Face compatible format, use convert_to_hf.py script:

```bash
usage: convert_to_hf.py [-h] [--model MODEL] [--config_path CONFIG_PATH] [--in_path_pt IN_PATH_PT] [--out_path OUT_PATH] [--save_safetensors] [--trust_remote_code] [--load_model] [--save_tokenizer]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to the model to base config on, as in AutoConfig.from_pretrained()
  --config_path CONFIG_PATH
                        Path to the model to base config on, as in AutoConfig.from_pretrained()
  --in_path_pt IN_PATH_PT
                        Path of the checkpoint to convert
  --out_path OUT_PATH   Path to save HF compatible checkpoint to
  --save_safetensors    Whether to save in safetensors format
  --trust_remote_code   Whether to trust remote code
  --load_model          Whether to load model
  --save_tokenizer      Whether to save tokenizer
```

# Benchmarks (matvec kernel)

In order to run the matvec benchmark suite, one should run:

```bash 
bench_spqr.py [-h] --tensor_path TENSOR_PATH [--ptcsr_path PTCSR_PATH] [--output_path OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  --tensor_path TENSOR_PATH
                        Path to folder containing the tensors of the formmodel_path/ 0/ tensor0 tensor1
  --ptcsr_path PTCSR_PATH
                        Path to folder containing the tensors of the formmodel_path/ 0/ tensor0 tensor1
  --output_path OUTPUT_PATH
                        Path to results *.csv file.

```

Make sure that the `<tensor_path>` and the optional `<ptcsr_path.` point to a folder containing quantized matrices produced by the `convert_legacy_model_format.py` script.
Use `<cuda_device_id>` to set the cuda device during benchmark. The script outputs the results in `<results_output>`.

# Tests

In order to run the unittest, simply execute:

```bash
python3 tests/test.py
```


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
