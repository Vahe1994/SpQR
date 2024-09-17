import unittest

import numpy as np
import torch
import time
import os
import sys

import inference

seed = 1
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')


def benchmark_ms(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res * 1000


class TestSparseFp16Fused(unittest.TestCase):
    def test_sparse_random(self):
        print('')
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device('cuda:0')
        for m in [16, 32, 64, 128, 256, 4096, 11008]:
            for n in [16, 32, 64, 128, 256, 4096, 11008]:
                for density in [0, 0.01, 0.05, 0.5, 0.9]:
                    for flag in [
                        # inference.FeatureFlag.SPARSE_NAIVE_FP16,
                        inference.FeatureFlag.SPARSE_FUSED_FP32,
                        # inference.FeatureFlag.SPARSE_MIXTURE_FP32,
                        #          inference.FeatureFlag.SPARSE_NAIVE_FP32
                    ]:
                        # inference.FeatureFlag.SPARSE_ALGO0_FP16, inference.FeatureFlag.SPARSE_ALGO0_FP32]:

                        # Generate test case
                        x_fp32 = inference.generate_x_fp32(n)
                        spqr_module, spqr_module_device = inference.create_random(m, n, density, device)

                        x_fp16_device = x_fp32.cuda(device=device).half()

                        deq_w = inference.spqr_dequantize_compressed(spqr_module).to(device)
                        deq_w_dense = inference.spqr_dequantize_dense_compressed(spqr_module).to(device)

                        y_true, _ = inference.torch_mul_timer(deq_w, x_fp16_device, 1)
                        y_true_dense, _ = inference.torch_mul_timer(deq_w, x_fp16_device, 1)
                        y = torch.zeros(m, dtype=torch.half, device=device)

                        inference.spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                        passed = torch.equal(y, y_true)

                        self.assertTrue(passed, msg=f'Failed for m = {m} n = {n} density = {density}\ny={y}\ny_true={y_true}')



class TestSparseFp16(unittest.TestCase):
    def test_sparse_random(self):
        print('')
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device('cuda:0')
        for m in [16, 32, 64, 128, 256, 4096, 11008]:
            for n in [16, 32, 64, 128, 256, 4096, 11008]:
                for density in [0, 0.01, 0.05, 0.5, 0.9]:
                    for flag in [
                        # inference.FeatureFlag.SPARSE_NAIVE_FP16,
                        inference.FeatureFlag.SPARSE_MIXTURE_FP32,
                        #          inference.FeatureFlag.SPARSE_NAIVE_FP32
                    ]:
                        # inference.FeatureFlag.SPARSE_ALGO0_FP16, inference.FeatureFlag.SPARSE_ALGO0_FP32]:
                        print(f'Running m = {m} n = {n} density = {density} flag = {flag}')

                        # Generate test case
                        x_fp32 = inference.generate_x_fp32(n)
                        spqr_module, spqr_module_device = inference.create_random(m, n, density, device)

                        x_fp16_device = x_fp32.cuda(device=device).half()

                        deq_w = inference.spqr_dequantize_compressed(spqr_module).to(device)
                        deq_w_dense = inference.spqr_dequantize_dense_compressed(spqr_module).to(device)

                        y_true, _ = inference.torch_mul_timer(deq_w, x_fp16_device, 1)
                        y_true_dense, _ = inference.torch_mul_timer(deq_w, x_fp16_device, 1)
                        y = torch.zeros(m, dtype=torch.half, device=device)

                        inference.spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                        passed = torch.equal(y, y_true)

                        print(f'{y}\n{y_true}')
                        self.assertTrue(passed)
                        print('OK')


if __name__ == '__main__':
    unittest.main()
