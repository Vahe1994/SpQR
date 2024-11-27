# Efficient SpQR Inference Kernel 


**Note:** This repository contains the single-batch inference kernel for a model quantizated via the SpQR algorithm
with a specific 16x16 tile and 3-bit configuration in mind, alsongside unstructured sparsity.
The compression algorithm is detailed in the research paper "[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)".


# Installation

### Packages

To install `spqr_quant`, run the following.
```bash
pip install -e .
```

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


# Inference

The file `inference_demo.py`  demos the functionality of this inference kernel in the context of
running end-to-end model inference. Below is a description of how to launch it.

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

This script also reports the mean median and minimimum time of the forward() passes and the total inference execution time.


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

