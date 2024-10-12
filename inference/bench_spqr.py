import numpy as np
import torch
import time
import os
import sys

import inference
import tests.test_util as test_util

from scipy.stats import gmean

if __name__ == '__main__':
    torch_runs = {}

    with open(sys.argv[3], 'w') as f:
        base_path = sys.argv[1]

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        NUM_RUNS = 50
        WARMUP = 1

        device = torch.device(f'cuda:{sys.argv[2]}')

        for m in [4096, 11008]:
            for n in [4096, 11008]:
                d = torch.zeros((m, n), dtype=torch.float16, device=device)
                x = torch.zeros(n, dtype=torch.float16, device=device)
                y, dense_runs = inference.torch_mul_timer(d, x, NUM_RUNS)
                this_algorithm = dense_runs[WARMUP:].min()
                torch_runs[(m, n)] = this_algorithm
                torch.cuda.empty_cache()
                time.sleep(2)


        folders = os.listdir(base_path)
        folders.sort()

        methods = [
            inference.FeatureFlag.SPARSE_FUSED_FP32_EXPERIMENTAL,
            inference.FeatureFlag.SPARSE_FUSED_FP32,
        ]

        f.write('Layer;Tensor Name;M;N;Sparsity (%)')

        for method in [inference.FeatureFlag.TORCH_FP16] + methods:
            f.write(f';{method.pretty()} (ms)')


        f.write('\n')

        benchmark_results_ms = []
        benchmark_speed_up = []

        for layer_id in folders:
            folder = os.path.join(base_path, layer_id)
            if not os.path.isdir(folder):
                continue
            for p in os.listdir(folder):
                tensor_path = os.path.join(folder, p)

                spqr_module: inference.SPQRModule = inference.load_compressed_tensor(tensor_path)

                m = spqr_module.m
                n = spqr_module.n
                print(f'Running {m} x {n}')

                deq_w = inference.spqr_dequantize_compressed(spqr_module)
                deq_w_dense = inference.spqr_dequantize_dense_compressed(spqr_module)

                spqr_module.to_device(device)
                spqr_module_device = spqr_module

                x_fp32 = test_util.generate_x_fp32(n)
                x_fp16_device = x_fp32.cuda(device=device).half()

                deq_w_device = deq_w.to(device).half().flatten()

                deq_w_device_dense = deq_w_dense.to(device).half().flatten()

                y_true, _ = inference.torch_mul_timer(deq_w_device, x_fp16_device, 1)
                y_true_dense, _ = inference.torch_mul_timer(deq_w_device_dense, x_fp16_device, 1)

                dense_speed_up = 0
                baseline_speed_up = 0

                sparsity_perc = spqr_module_device.sparsity * 100

                torch_run = torch_runs[(spqr_module_device.m, spqr_module_device.n)]

                f.write(f'{layer_id};{p};{m};{n};{sparsity_perc:.3f};{torch_run:.4f}')

                for flag in methods:
                    print(f'Running {repr(flag)} on {layer_id}.{p}')
                    

                    y, spqr_runs = inference.spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS)

                    spqr_runs = spqr_runs[WARMUP:]
                    this_algorithm = spqr_runs.min()


                    speed_up = torch_run / this_algorithm
                    print(
                        f'\t{repr(flag)} running {this_algorithm} ms {speed_up :.2f}X speed-up')

                    f.write(f';{this_algorithm:.4f}')

                    if flag == inference.FeatureFlag.DENSE_ONLY_FP16:
                        dense_speed_up = speed_up
                    elif flag == inference.FeatureFlag.SPARSE_MIXTURE_FP32 or \
                         flag == inference.FeatureFlag.SPARSE_FUSED_FP32:
                        baseline_speed_up = speed_up
                        benchmark_results_ms.append(this_algorithm)
                        benchmark_speed_up.append(baseline_speed_up)

                    if flag == inference.FeatureFlag.DENSE_ONLY_FP16 or flag == inference.FeatureFlag.DENSE_ONLY_FP32:
                        assert (torch.allclose(y, y_true_dense, atol=1e-1, rtol=1e-1))

                f.write('\n')
                f.flush()
                print('\n\n')

            print(f'Total benchmark geomean = {gmean(benchmark_results_ms)}')
            print(f'Total benchmark speed-up geomean = {gmean(benchmark_speed_up)}')

            print(f'Total benchmark mean = {np.array(benchmark_results_ms).mean()}')
            print(f'Total benchmark speed-up mean= {np.array(benchmark_speed_up).mean()}')

            print('\n\n')
