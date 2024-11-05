import unittest

import numpy as np
import torch
import spqr_cuda

import test_util
import inference
from inference import SparseStorageConfiguration

seed = 1
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')

def spqr_mul(spqr_device: inference.QuantizedLinear, x, y, feature_flag: inference.FeatureFlags):
    spqr_cuda.spqr_mul(
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
        int(feature_flag)
    )


def torch_mul(deq_w, x):
    runs = torch.empty(1).cpu().float()

    if len(deq_w.shape) == 1:
        n = x.shape[0]
        m = deq_w.shape[0] // n
    else:
        m, n = deq_w.shape

    assert (n == x.shape[0])

    y = torch.zeros(m, dtype=x.dtype, device=x.device)

    spqr_cuda.torch_mul_fp16(m, n, deq_w, x, y, runs[0])

    return y


class TestSparseFp16Easy(unittest.TestCase):
    def test_sparse_ones(self):
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device('cuda:0')
        for m in [16, 32, 64, 128, 256]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for density in [0]:  # , 0.01, 0.05, 0.5, 0.9]:
                    for flag in [
                        inference.FeatureFlags.SPARSE_FUSED_FP32,
                    ]:
                        print(f'Running m = {m} n = {n}')
                        # Generate test case
                        x_fp32 = test_util.generate_x_fp32(n) * 0 + 1
                        spqr_module, spqr_module_device = test_util.create_ones(m, n)

                        x_fp16_device = x_fp32.cuda(device=device).half()

                        deq_w = spqr_module.dequantize().to(device)

                        y_true = torch_mul(deq_w, x_fp16_device)
                        y = torch.zeros(m, dtype=torch.half, device=device)

                        spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                        passed = torch.equal(y, y_true)

                        self.assertTrue(passed,
                                        msg=f'Failed for m = {m} n = {n} density = {density}\ny={y}\ny_true={y_true}')


class TestSparseFp16DenseOnly(unittest.TestCase):
    def test_sparse_random(self):
        print('')
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device('cuda:0')
        for m in [16, 32, 64, 128, 256]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for create_matrix in [test_util.create_ones, test_util.create_random_weights_ones,
                                      test_util.create_random_first_order_ones,
                                      test_util.create_random_second_order_ones, test_util.create_random]:
                    for create_x in [test_util.create_x_zeros, test_util.create_x_ones, test_util.create_x_random]:
                        for flag in [inference.FeatureFlags.SPARSE_FUSED_FP32]:
                            # Generate test case
                            x_fp16_device = create_x(n).cuda(device=device)
                            spqr_module, quantized_linear = create_matrix(m, n, 0)

                            deq_w = spqr_module.dequantize().to(device)

                            y_true = torch_mul(deq_w, x_fp16_device)
                            y = torch.zeros(m, dtype=torch.half, device=device)

                            spqr_mul(quantized_linear, x_fp16_device, y, flag)

                            passed = torch.equal(y, y_true)

                            self.assertTrue(passed,
                                            msg=f'\n\n\nFailed for m = {m} n = {n} density = {0}\ny={y}\ny_true={y_true}\nmatrix method={create_matrix}\nx method={create_x}')


class TestSparseFp16Fused(unittest.TestCase):
    def test_sparse_random(self):
        print('')
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device('cuda:0')
        for m in [16, 32, 64, 128, 256, 512, 4096, 11008]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for density in [0, 0.01, 0.05, 0.5, 0.9]:
                    for compression_strategy in [SparseStorageConfiguration.CSR, SparseStorageConfiguration.PTCSR]:
                        for flag in [
                            inference.FeatureFlags.SPARSE_FUSED_FP32,
                        ]:
                            # Generate test case
                            x_fp32 = test_util.generate_x_fp32(n)
                            spqr_module, spqr_module_device = test_util.create_random(m, n, density,
                                                                                      compression_strategy)

                            x_fp16_device = x_fp32.cuda(device=device).half()

                            deq_w = spqr_module.dequantize().to(device)

                            y_true = torch_mul(deq_w, x_fp16_device)
                            y = torch.zeros(m, dtype=torch.half, device=device)

                            spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                            passed = torch.equal(y, y_true)

                            self.assertTrue(passed,
                                            msg=f'Failed for m = {m} n = {n} density = {density} compression_strategy = {compression_strategy}\ny={y}\ny_true={y_true}')


if __name__ == '__main__':
    unittest.main()
