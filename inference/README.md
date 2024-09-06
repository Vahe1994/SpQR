# Building the Repository

You can build this repository by running the following command:

```bash
/bin/bash scripts/build.sh
```

# Conversion Script

In order the SpMV performance of tensors from a model produced by the SpQR algorithm, run:

```bash
python3 compress.py <path_to_spqr_model_output> <compressed_model_path>
```

# Running the Benchmark

In order to run the benchmark suite, one should run:

```bash
python3 bench_spqr.py <path_to_data>
```
 
Make sure that the `<path_to_data>` points to a folder containing quantized matrices produced by the `convert.py` script. 

# Tests


```bash
python3 tests/test.py
```
