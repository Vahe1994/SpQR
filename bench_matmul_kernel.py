# /home/elvircrn/CLionProjects/cutlass/build/tools/profiler/cutlass_profiler --operation=Gemm --m=4096,11008 --n=4096,11008 --k=1,2,4,8 --A=f16:column --B=f16:row --C=f16:column --accum=f32 --profiling-iterations=2000 --stages=3 --warmup-iterations=100
import warnings
warnings.filterwarnings("ignore")

import io

import numpy as np
import pandas as pd
import torch

# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# torch.backends.cudnn.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")


from scipy.stats import gmean
from spqr_quant import QuantizedLinear
from spqr_quant.inference import FeatureFlags
from spqr_quant.inference_kernels.kernel_selector import (
    get_spqr_mul_timer_batched,
)

from spqr_quant.inference import SparseStorageConfiguration
from tests.test import create_random, generate_x_fp32

cutlass_str = """
m,n,k,Runtime
4096,4096,1,0.139133
4096,4096,2,0.13965
4096,4096,4,0.139508
4096,4096,8,0.140223
4096,4096,16,0.141828
4096,11008,1,0.366857
4096,11008,2,0.368438
4096,11008,4,0.369755
4096,11008,8,0.372519
4096,11008,16,0.379609
11008,4096,1,0.368447
11008,4096,2,0.370071
11008,4096,4,0.371965
11008,4096,8,0.379108
11008,4096,16,0.389834
11008,11008,1,0.985347
11008,11008,2,0.985883
11008,11008,4,0.988083
11008,11008,8,1.00129
11008,11008,16,1.02741
16384,16384,1,2.16175
16384,16384,2,2.16977
16384,16384,4,2.18473
16384,16384,8,2.19353
"""

cutlass_data = io.StringIO(cutlass_str)
cutlass_runs = pd.read_csv(cutlass_data)


def spqr_mul_timer(spqr_device: QuantizedLinear, x, feature_flag: FeatureFlags, num_runs, k):
    runs = torch.empty(num_runs).cpu().float()
    y = torch.zeros(spqr_device.m * k, dtype=x.dtype, device=x.device).flatten().contiguous()

    for i in range(num_runs):
        y = torch.zeros_like(y)
        get_spqr_mul_timer_batched()(
            spqr_device.m,
            spqr_device.n,
            k,
            spqr_device.bits,
            spqr_device.beta1,
            spqr_device.beta2,
            spqr_device.dense_weights,
            spqr_device.row_offsets,
            spqr_device.col_vals,
            spqr_device.nnz,
            x,
            y,
            runs[i],
            feature_flag,
        )

    return y, runs


def torch_mul_timer_runs(deq_w, x, num_runs, k):
    if k == -1:
        if len(deq_w.shape) == 1:
            n = x.shape[0]
            m = deq_w.shape[0] // n
        else:
            m, n = deq_w.shape

        assert n == x.shape[0]

        runs = torch.empty(num_runs).cpu().float()

        y = torch.zeros(m, dtype=x.dtype, device=x.device)

        for i in range(num_runs):
            y = torch.zeros_like(y)
            get_torch_mul_timer()(deq_w, x, y, runs[i])

        return y, runs
    else:
        if len(deq_w.shape) == 1:
            n = x.shape[0]
            m = deq_w.shape[0] // n
        else:
            m, n = deq_w.shape

        assert n == x.shape[0]

        runs = torch.empty(num_runs).cpu().float()

        y = torch.zeros((m, k), dtype=x.dtype, device=x.device).contiguous()

        for i in range(num_runs):
            y = torch.zeros_like(y)
            get_torch_mul_timer_batched()(deq_w, x, y, runs[i])

        return y, runs


def bench_random():
    m = 2 ** 14
    n = 2 ** 14
    NUM_RUNS = 2000
    device = 'cuda'
    flag = FeatureFlags.SPARSE_FUSED_FP32

    print('m,n,k,Density,Sparse Storage,Speed-Up (X)')
    for p in [SparseStorageConfiguration.CSR,
              SparseStorageConfiguration.PTCSR]:
        for d in [0., 0.01, 0.02, 0.03]:
            for k in [1, 2, 4, 8]:
                seed = 1337
                np.random.seed(seed)
                torch.random.manual_seed(seed)

                x_fp32 = generate_x_fp32(n * k)
                spqr_module, spqr_module_device = create_random(
                    m, n, d, p
                )
                x_fp16_device = x_fp32.cuda(device=device).half()

                y_csr, spqr_runs = spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS, k)

                cutlass_run = cutlass_runs[
                    (cutlass_runs["m"] == m) &
                    (cutlass_runs["n"] == n) &
                    (cutlass_runs["k"] == k)]["Runtime"].item()
                print(f'{m},{n},{k},{d},{p},{cutlass_run / spqr_runs.min():.3f}')


if __name__ == '__main__':
    bench_random()
#
# if False:
#     if __name__ == "__main__":
#         torch_runs = {}
#
#         parser = argparse.ArgumentParser(add_help=True)
#
#         parser.add_argument(
#             "--tensor_path",
#             type=str,
#             required=True,
#             help="Path to folder containing the tensors of the form"
#             "model_path/"
#             "   0/"
#             "       tensor0"
#             "       tensor1",
#         )
#
#         parser.add_argument(
#             "--ptcsr_path",
#             type=str,
#             required=False,
#             help="Path to folder containing the tensors of the form"
#             "model_path/"
#             "   0/"
#             "       tensor0"
#             "       tensor1",
#         )
#
#         parser.add_argument(
#             "--output_path",
#             type=str,
#             help="Path to results *.csv file.",
#         )
#
#         args = parser.parse_args()
#
#         with open(args.output_path, "w") as f:
#             run_ptcsr = args.ptcsr_path is not None
#
#             base_path = args.tensor_path
#             base_path_modified_csr = args.ptcsr_path
#
#             seed = 1
#             np.random.seed(seed)
#             torch.random.manual_seed(seed)
#
#             NUM_RUNS = 2000
#             WARMUP = 10
#
#             device = torch.device("cuda")
#
#             csr_folders = os.listdir(base_path)
#
#             if run_ptcsr:
#                 folders_modified_csr = os.listdir(base_path_modified_csr)
#             else:
#                 folders_modified_csr = os.listdir(base_path)
#
#             csr_folders.sort()
#             folders_modified_csr.sort()
#
#             methods = [
#                 FeatureFlags.SPARSE_FUSED_FP32,
#             ]
#
#             f.write("Layer;Tensor Name;M;N;Sparsity (%)")
#
#             for method in [FeatureFlags.TORCH_FP16] + methods:
#                 f.write(f";{method.pretty()} (ms)")
#
#             f.write(f";{method.pretty()} Modified CSR (ms)")
#
#             f.write("\n")
#
#             benchmark_results_ms = []
#             benchmark_speed_up = []
#
#             for layer_id in csr_folders:
#                 folder = os.path.join(base_path, layer_id)
#                 folder_ptcsr = os.path.join(base_path_modified_csr, layer_id)
#
#                 if run_ptcsr:
#                     folders_modified_csr = os.path.join(base_path_modified_csr, layer_id)
#                 else:
#                     folders_modified_csr = os.path.join(base_path, layer_id)
#                 if not os.path.isdir(folder):
#                     continue
#
#                 for p, p_modified_csr in zip(os.listdir(folder), os.listdir(folder_ptcsr)):
#                     for k in [1, 2, 4, 8]:
#                         tensor_path = os.path.join(folder, p)
#                         tensor_path_modified_csr = os.path.join(folder_ptcsr, p_modified_csr)
#
#                         spqr_module_modified_csr = torch.load(tensor_path_modified_csr)
#
#                         deq_w_modified_csr = spqr_module_modified_csr.dequantize()
#
#                         spqr_module_modified_csr.to(device=device)
#                         spqr_module_device_modified_csr = spqr_module_modified_csr
#
#                         spqr_module = torch.load(tensor_path)
#
#                         m = spqr_module.m
#                         n = spqr_module.n
#                         print(f"Running {m} x {n} x {k}")
#
#                         cutlass_run = cutlass_runs[
#                             (cutlass_runs["m"] == m) & (cutlass_runs["n"] == n) & (cutlass_runs["k"] == k)
#                         ]["Runtime"].item()
#                         deq_w = spqr_module.dequantize()
#
#                         spqr_module.to(device=device)
#                         spqr_module_device = spqr_module
#
#                         def generate_x_fp32(n, upper_bound=3):
#                             x_fp32 = ((torch.rand(n) - 0.5) * 4 * upper_bound).int()
#                             return x_fp32.float()
#
#                         x_fp32 = generate_x_fp32(k * n)
#                         x_fp16_device = x_fp32.cuda(device=device).half().contiguous()
#
#                         deq_w_device = deq_w.to(device).half().flatten()
#
#                         dense_speed_up = 0
#                         baseline_speed_up = 0
#
#                         sparsity_perc = spqr_module_device.sparsity * 100
#
#                         f.write(f"{layer_id};{p};{m};{n};{sparsity_perc:.3f};{cutlass_run:.4f}")
#
#                         for flag in methods:
#                             print(f"Running {repr(flag)} on {layer_id}.{p}")
#
#                             y_csr, spqr_runs = spqr_mul_timer(spqr_module_device, x_fp16_device, flag, NUM_RUNS, k)
#                             spqr_runs = spqr_runs[WARMUP:]
#                             this_algorithm = spqr_runs.min()
#
#                             y_ptcsr, spqr_runs_modified_csr = spqr_mul_timer(
#                                 spqr_module_device_modified_csr, x_fp16_device, flag, NUM_RUNS, k
#                             )
#                             # assert torch.allclose(y_csr, y_ptcsr)
#
#                             spqr_runs_modified_csr = spqr_runs_modified_csr[WARMUP:]
#
#                             speed_up = cutlass_run / this_algorithm
#
#                             print(
#                                 f"\t{repr(flag)} running {this_algorithm} ms {speed_up:.2f}X speed-up vs torch {cutlass_run} ms"
#                             )
#
#                             if run_ptcsr:
#                                 this_algorithm_modified_csr = spqr_runs_modified_csr.min()
#                                 speed_up_modified_csr = cutlass_run / this_algorithm_modified_csr
#                                 print(
#                                     f"\t{repr(flag)} modified csr running {this_algorithm_modified_csr} ms {speed_up_modified_csr:.2f}X speed-up vs torch {cutlass_run} ms"
#                                 )
#
#                             if run_ptcsr:
#                                 f.write(f";{this_algorithm:.4f};{this_algorithm_modified_csr:.4f}")
#                                 baseline_speed_up = max(speed_up, speed_up_modified_csr)
#                             else:
#                                 baseline_speed_up = speed_up
#                                 f.write(f";{this_algorithm:.4f}")
#
#                             if run_ptcsr:
#                                 benchmark_results_ms.append(min(this_algorithm, this_algorithm_modified_csr))
#                             else:
#                                 benchmark_results_ms.append(this_algorithm)
#                             benchmark_speed_up.append(baseline_speed_up)
#
#                         f.write("\n")
#                         f.flush()
#                         print("\n\n")
#
#                 print(f"Total benchmark geomean = {gmean(benchmark_results_ms)}")
#                 print(f"Total benchmark speed-up geomean = {gmean(benchmark_speed_up)}")
#
#                 print(f"Total benchmark mean = {np.array(benchmark_results_ms).mean()}")
#                 print(f"Total benchmark speed-up mean= {np.array(benchmark_speed_up).mean()}")
#
#                 print("\n\n")
