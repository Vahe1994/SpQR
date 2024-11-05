import numpy as np
import torch
import spqr_cuda
import time
import os
import sys

import inference
import tests.test_util as test_util

from scipy.stats import gmean


def spqr_mul_timer(spqr_device: inference.QuantizedLinear, x, feature_flag: inference.FeatureFlags, num_runs):
    runs = torch.empty(num_runs).cpu().float()
    y = torch.zeros(spqr_device.m, dtype=x.dtype, device=x.device)

    for i in range(num_runs):
        y = torch.zeros_like(y)
        spqr_cuda.spqr_mul_timer(
            spqr_device.m,
            spqr_device.n,
            spqr_device.bits,
            spqr_device.beta1,
            spqr_device.beta2,
            spqr_device.buff0,
            spqr_device.row_offsets,
            spqr_device.col_vals,
            spqr_device.nnz,
            x,
            y,
            runs[i],
            feature_flag)

    return y, runs


def torch_mul_timer(deq_w, x, num_runs):
    if len(deq_w.shape) == 1:
        n = x.shape[0]
        m = deq_w.shape[0] // n
    else:
        m, n = deq_w.shape

    assert (n == x.shape[0])

    runs = torch.empty(num_runs).cpu().float()

    y = torch.zeros(m, dtype=x.dtype, device=x.device)

    for i in range(num_runs):
        y = torch.zeros_like(y)
        spqr_cuda.torch_mul_fp16(m, n, deq_w, x, y, runs[i])

    return y, runs


if __name__ == '__main__':
    torch_runs = {}

    with open(sys.argv[3], 'w') as f:
        base_path = sys.argv[1]
        base_path_modified_csr = f'{sys.argv[1]}_ptcsr'

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        NUM_RUNS = 2000
        WARMUP = 10

        device = torch.device(f'cuda:{sys.argv[2]}')

        for m in [4096, 11008]:
            for n in [4096, 11008]:
                d = torch.zeros((m, n), dtype=torch.float16, device=device)
                x = torch.zeros(n, dtype=torch.float16, device=device)
                y, dense_runs = torch_mul_timer(d, x, NUM_RUNS)
                this_algorithm = dense_runs[WARMUP:].min()
                torch_runs[(m, n)] = this_algorithm
                torch.cuda.empty_cache()
                time.sleep(2)

        folders = os.listdir(base_path)
        folders_modified_csr = os.listdir(base_path_modified_csr)
        folders.sort()

        methods = [
            inference.FeatureFlags.SPARSE_FUSED_FP32,
        ]

        f.write('Layer;Tensor Name;M;N;Sparsity (%)')

        for method in [inference.FeatureFlags.TORCH_FP16] + methods:
            f.write(f';{method.pretty()} (ms)')

        f.write(f';{method.pretty()} Modified CSR (ms)')

        f.write('\n')

        benchmark_results_ms = []
        benchmark_speed_up = []

        for layer_id in folders:
            folder = os.path.join(base_path, layer_id)
            folders_modified_csr = os.path.join(base_path_modified_csr, layer_id)
            if not os.path.isdir(folder):
                continue
            for p, p_modified_csr in zip(os.listdir(folder), os.listdir(folders_modified_csr)):
                tensor_path = os.path.join(folder, p)
                tensor_path_modified_csr = os.path.join(folder, p_modified_csr)

                spqr_module = torch.load(tensor_path)
                spqr_module_modified_csr = torch.load(tensor_path_modified_csr)

                m = spqr_module.m
                n = spqr_module.n
                print(f'Running {m} x {n}')

                deq_w = spqr_module.dequantize()

                spqr_module.to(device=device)
                spqr_module_modified_csr.to(device=device)
                spqr_module_device = spqr_module
                spqr_module_device_modified_csr = spqr_module_modified_csr

                x_fp32 = test_util.generate_x_fp32(n)
                x_fp16_device = x_fp32.cuda(device=device).half()

                deq_w_device = deq_w.to(device).half().flatten()

                dense_speed_up = 0
                baseline_speed_up = 0

                sparsity_perc = spqr_module_device.sparsity * 100

                torch_run = torch_runs[(spqr_module_device.m, spqr_module_device.n)]

                f.write(f'{layer_id};{p};{m};{n};{sparsity_perc:.3f};{torch_run:.4f}')

                for flag in methods:
                    print(f'Running {repr(flag)} on {layer_id}.{p}')

                    y, spqr_runs = spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS)
                    spqr_runs = spqr_runs[WARMUP:]
                    this_algorithm = spqr_runs.min()

                    torch.cuda.empty_cache()
                    time.sleep(1)

                    y, spqr_runs_modified_csr = spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS)
                    spqr_runs_modified_csr = spqr_runs_modified_csr[WARMUP:]
                    this_algorithm_modified_csr = spqr_runs_modified_csr.min()

                    speed_up = torch_run / this_algorithm
                    speed_up_modified_csr = torch_run / this_algorithm_modified_csr

                    print(
                        f'\t{repr(flag)} running {this_algorithm} ms {speed_up :.2f}X speed-up vs torch {torch_run} ms')
                    print(
                        f'\t{repr(flag)} modified csr running {this_algorithm_modified_csr} ms {speed_up_modified_csr :.2f}X speed-up vs torch {torch_run} ms')

                    f.write(f';{this_algorithm:.4f};{this_algorithm_modified_csr:.4f}')

                    baseline_speed_up = max(speed_up, speed_up_modified_csr)
                    benchmark_results_ms.append(min(this_algorithm, this_algorithm_modified_csr))
                    benchmark_speed_up.append(baseline_speed_up)

                f.write('\n')
                f.flush()
                print('\n\n')

            print(f'Total benchmark geomean = {gmean(benchmark_results_ms)}')
            print(f'Total benchmark speed-up geomean = {gmean(benchmark_speed_up)}')

            print(f'Total benchmark mean = {np.array(benchmark_results_ms).mean()}')
            print(f'Total benchmark speed-up mean= {np.array(benchmark_speed_up).mean()}')

            print('\n\n')
