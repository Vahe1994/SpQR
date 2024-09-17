import numpy as np
import torch
import time
import os
import sys

import inference

from scipy.stats import gmean

if __name__ == '__main__':
    with open('report/v2_results_rtx_4060.csv', 'w') as f:
        base_path = sys.argv[1]

        seed = 1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        NUM_RUNS = 25
        WARMUP = 10

        device = torch.device(f'cuda:{sys.argv[2]}')

        folders = os.listdir(base_path)
        folders.sort()

        methods = [
            # inference.FeatureFlag.SPARSE_SHARED_FP16,
            # inference.FeatureFlag.SPARSE_SHARED_BASELINE_FP16,
            # inference.FeatureFlag.SPARSE_FUSED_FP16,
            # inference.FeatureFlag.SPARSE_SINGLE_FP16,
            # inference.FeatureFlag.SPARSE_CUSPARSE_FP16,
            # inference.FeatureFlag.DENSE_ONLY_FP16,
            inference.FeatureFlag.TORCH_FP16,
            # inference.FeatureFlag.DENSE_ONLY_FP32,
            # inference.FeatureFlag.SPARSE_MIXTURE_FP32,
            inference.FeatureFlag.SPARSE_FUSED_FP32,
        ]

        f.write('Layer;Tensor Name;M;N;Sparsity (%)')

        for method in methods:
            f.write(f';{method.pretty()} (ms)')

        f.write('\n')

        benchmark_results_ms = []
        benchmark_speed_up = []

        for layer_id in folders:
            folder = os.path.join(base_path, layer_id)
            if not os.path.isdir(folder):
                continue
            for p in os.listdir(folder):
                torch.cuda.empty_cache()
                time.sleep(1)

                tensor_path = os.path.join(folder, p)

                spqr_module: inference.SPQRModule = inference.load_compressed_tensor(tensor_path)

                m = spqr_module.m
                n = spqr_module.n
                print(f'Running {m} x {n}')

                deq_w = inference.spqr_dequantize_compressed(spqr_module)
                deq_w_dense = inference.spqr_dequantize_dense_compressed(spqr_module)

                spqr_module.to_device(device)
                spqr_module_device = spqr_module

                x_fp32 = inference.generate_x_fp32(n)
                x_fp16_device = x_fp32.cuda(device=device).half()

                deq_w_device = deq_w.to(device).half().flatten()

                deq_w_device_dense = deq_w_dense.to(device).half().flatten()

                y_true, _ = inference.torch_mul_timer(deq_w_device, x_fp16_device, 1)
                y_true_dense, _ = inference.torch_mul_timer(deq_w_device_dense, x_fp16_device, 1)

                dense_speed_up = 0
                baseline_speed_up = 0

                sparsity_perc = spqr_module_device.sparsity * 100

                torch_run = 0


                f.write(f'Layer {layer_id};{p};{m};{n};{sparsity_perc:.2f}')

                with torch.no_grad():
                    for flag in methods:
                        torch.cuda.empty_cache()
                        time.sleep(1)
                        print(f'Running {repr(flag)} on {layer_id}.{p}')

                        if flag == inference.FeatureFlag.TORCH_FP16:
                            y, spqr_runs = inference.torch_mul_timer(deq_w_device, x_fp16_device, NUM_RUNS)
                        else:
                            y, spqr_runs = inference.spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS)

                        spqr_runs = spqr_runs[WARMUP:]
                        this_algorithm = spqr_runs.min()

                        if flag == inference.FeatureFlag.TORCH_FP16:
                            torch_run = this_algorithm

                        speed_up = torch_run / this_algorithm
                        print(
                            f'\t{repr(flag)} running {this_algorithm} ms {speed_up :.2f}X speed-up')

                        f.write(f';{this_algorithm:.2f}')

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
