import unittest
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from inference_lib.spqr_quant import QuantizedLinear
from inference_lib.spqr_quant.inference import FeatureFlags, ModelArgs, SparseStorageConfiguration, SPQRLegacy, updiv
from inference_lib.spqr_quant.inference_kernels.kernel_selector import get_torch_mul_timer, \
    get_spqr_mul_fused, get_spqr_mul


def generate_x_fp32(n, upper_bound=3):
    x_fp32 = ((torch.rand(n) - 0.5) * 4 * upper_bound).int()
    return x_fp32.float()


def create_x_random(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half()


def create_x_ones(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half() * 0 + 1


def create_x_zeros(n, upper_bound=3):
    return generate_x_fp32(n, upper_bound).half() * 0


def random_like(v: torch.Tensor, upper_bound=1):
    return generate_x_fp32(v.shape[0], upper_bound).to(dtype=v.dtype, device=v.device)


def generate_3bit(n):
    upper_bound = 3
    x_char = (torch.rand(n) * upper_bound).char()
    return x_char


def generate_x_int32(n):
    upper_bound = 3
    x_int32 = ((torch.rand(n) - 0.5) * upper_bound).int()
    return x_int32


def random_csr_host(m, n, density):
    r = ((torch.rand(m, n) <= density) * (torch.ones(m, n) * 1).int()).to_sparse_csr()

    return r.crow_indices().int(), r.values().half(), r.col_indices().short(), r._nnz()


@dataclass
class DenseWeightInitStrategy:
    randomize: bool = False
    set_all: torch.float16 = None
    arange: bool = False


@dataclass
class SparseWeightInitStrategy:
    sparsity: float = 0.0


@dataclass
class MatrixBuilder:
    m: int
    n: int
    weights: DenseWeightInitStrategy
    first_order: DenseWeightInitStrategy
    second_order: DenseWeightInitStrategy


def create_spqr_quantized_matrix(
    m: int,
    n: int,
    weight_init_strategy: int = None,
    first_order_init_strategy: int = None,
    second_order_init_strategy: torch.float16 = None,
    density: float = 0.0,
    sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR,
    in_perm=None,
) -> Tuple[QuantizedLinear, QuantizedLinear]:
    beta1, beta2 = 16, 16

    if weight_init_strategy is None:
        W_quantized = generate_3bit(m * n)
        W = W_quantized.char()
    else:
        W = (torch.ones(m * n) * weight_init_strategy).char()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    if first_order_init_strategy is None:
        W_s_raw = generate_3bit(num_first_order_groups)
        W_z_raw = generate_3bit(num_first_order_groups)
        W_s = W_s_raw.char()
        W_z = W_z_raw.char()
    else:
        W_s = (torch.ones(m * n) * first_order_init_strategy).char()
        W_z = torch.zeros(m * n).char()

    if second_order_init_strategy is None:
        W_s_s = generate_x_fp32(num_second_order_groups).half()
        W_s_z = generate_x_fp32(num_second_order_groups).half()
        W_z_s = generate_x_fp32(num_second_order_groups).half()
        W_z_z = generate_x_fp32(num_second_order_groups).half()
    else:
        W_s_s = torch.ones(num_second_order_groups).half() * second_order_init_strategy
        W_s_z = torch.zeros(num_second_order_groups).half()
        W_z_s = torch.ones(num_second_order_groups).half() * second_order_init_strategy
        W_z_z = torch.zeros(num_second_order_groups).half()

    if density == 0:
        values = torch.zeros(0).half()
        row_offsets = torch.zeros(m + 1).int()
        col_ids = torch.zeros(0).short()
    else:
        row_offsets, values, col_ids, nnz = random_csr_host(m, n, density)

    spqr_legacy = SPQRLegacy(
        m, n, 3, W, 16, 16, W_s, W_z, W_s_s, W_s_z, W_z_s, W_z_z, row_offsets, col_ids, values, in_perm
    )

    mod = QuantizedLinear.from_legacy(spqr_legacy, ModelArgs(3, 16, 16, sparse_storage), "cpu")
    mod_device = QuantizedLinear.from_legacy(spqr_legacy, ModelArgs(3, 16, 16, sparse_storage), "cuda")

    return mod, mod_device


def create_ones(m, n, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, 1, 1, 1, 0.0, sparse_storage, None)


def create_random(m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, None, None, None, density, sparse_storage, None)


def create_random_weights_ones(
    m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR
):
    return create_spqr_quantized_matrix(m, n, 1, None, None, density, sparse_storage, None)


def create_random_first_order_ones(
    m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR
):
    return create_spqr_quantized_matrix(m, n, None, 1, None, density, sparse_storage, None)


def create_random_second_order_ones(
    m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR
):
    return create_spqr_quantized_matrix(m, n, None, None, 1, density, sparse_storage, None)


def create_just_sparse(m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, 0, 0, 0, density, sparse_storage, None)


seed = 1
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device("cuda:0")

def _spqr_mul_fused(spqr_device: QuantizedLinear, x, y, feature_flag: FeatureFlags):
    get_spqr_mul_fused()(
        spqr_device.m,
        spqr_device.n,
        spqr_device.bits,
        spqr_device.beta1,
        spqr_device.beta2,
        torch.arange(spqr_device.n, dtype=torch.short, device=x.device),
        spqr_device.dense_weights,
        spqr_device.row_offsets,
        spqr_device.col_vals,
        spqr_device.nnz,
        x,
        int(feature_flag),
        y,
        y,
    )


def _spqr_mul(spqr_device: QuantizedLinear, x, y, feature_flag: FeatureFlags):
    get_spqr_mul()(
        spqr_device.m,
        spqr_device.n,
        spqr_device.bits,
        spqr_device.beta1,
        spqr_device.beta2,
        spqr_device.dense_weights,
        spqr_device.row_offsets,
        spqr_device.col_vals,
        spqr_device.nnz,
        x,
        int(feature_flag),
        y,
        y,
    )


def torch_mul(deq_w, x):
    runs = torch.empty(1).cpu().float()

    if len(deq_w.shape) == 1:
        n = x.shape[0]
        m = deq_w.shape[0] // n
    else:
        m, n = deq_w.shape

    assert n == x.shape[0]

    y = torch.zeros(m, dtype=x.dtype, device=x.device)

    get_torch_mul_timer()(deq_w, x, y, runs[0])

    return y


class TestSparseFp16Easy(unittest.TestCase):
    def test_sparse_ones(self):
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")
        for m in [16, 32, 64, 128, 256]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for density in [0]:  # , 0.01, 0.05, 0.5, 0.9]:
                    for flag in [
                        FeatureFlags.SPARSE_FUSED_FP32,
                    ]:
                        print(f"Running m = {m} n = {n}")
                        # Generate test case
                        x_fp32 = generate_x_fp32(n) * 0 + 1
                        spqr_module, spqr_module_device = create_ones(m, n)

                        x_fp16_device = x_fp32.cuda(device=device).half()

                        deq_w = spqr_module.dequantize().to(device)

                        y_true = torch_mul(deq_w, x_fp16_device)
                        y = torch.zeros(m, dtype=torch.half, device=device)

                        _spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                        passed = torch.equal(y, y_true)

                        self.assertTrue(
                            passed, msg=f"Failed for m = {m} n = {n} density = {density}\ny={y}\ny_true={y_true}"
                        )


class TestSparseFp16DenseOnly(unittest.TestCase):
    def test_dense_random(self):
        print("")
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")
        for m in [16, 32, 64, 128, 256]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for create_matrix in [
                    create_ones,
                    create_random_weights_ones,
                    create_random_first_order_ones,
                    create_random_second_order_ones,
                    create_random,
                ]:
                    for create_x in [create_x_zeros, create_x_ones, create_x_random]:
                        for flag in [FeatureFlags.SPARSE_FUSED_FP32]:
                            # Generate test case
                            x_fp16_device = create_x(n).cuda(device=device)
                            spqr_module, quantized_linear = create_random(m, n, 0)

                            deq_w = spqr_module.dequantize().to(device)

                            y_true = torch_mul(deq_w, x_fp16_device)
                            y = torch.zeros(m, dtype=torch.half, device=device)

                            _spqr_mul(quantized_linear, x_fp16_device, y, flag)

                            passed = torch.equal(y, y_true)

                            self.assertTrue(
                                passed,
                                msg=f"\n\n\nFailed for m = {m} n = {n} density = {0}\ny={y}\ny_true={y_true}\nmatrix method={create_matrix}\nx method={create_x}",
                            )


class TestSparseFp16Fused(unittest.TestCase):
    def test_sparse_random(self):
        print("")
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")
        for m in [16, 32, 64, 128, 256, 512, 4096, 11008]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for density in [0, 0.01, 0.05, 0.5, 0.9]:
                    for compression_strategy in [SparseStorageConfiguration.CSR, SparseStorageConfiguration.PTCSR]:
                        for flag in [
                            FeatureFlags.SPARSE_FUSED_FP32,
                        ]:
                            print(f"Running m = {m} n = {n} density = {density} storage = {compression_strategy}")
                            # Generate test case
                            x_fp32 = generate_x_fp32(n)
                            spqr_module, spqr_module_device = create_random(m, n, density, compression_strategy)

                            x_fp16_device = x_fp32.cuda(device=device).half()

                            deq_w = spqr_module.dequantize().to(device)

                            y_true = torch_mul(deq_w, x_fp16_device)
                            y = torch.zeros(m, dtype=torch.half, device=device)

                            _spqr_mul(spqr_module_device, x_fp16_device, y, flag)

                            passed = torch.equal(y, y_true)

                            self.assertTrue(
                                passed,
                                msg=f"Failed for m = {m} n = {n} density = {density} compression_strategy = {compression_strategy}\ny={y}\ny_true={y_true}",
                            )


class TestSparseFp16FusedFused(unittest.TestCase):
    def test_sparse_random(self):
        print("")
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")
        for m in [16, 32, 64, 128, 256]:
            for n in [16, 32, 64, 128, 256, 512, 4096, 11008]:
                for density in [0, 0.01, 0.05, 0.5, 0.9]:
                    for compression_strategy in [SparseStorageConfiguration.CSR, SparseStorageConfiguration.PTCSR]:
                        for flag in [
                            FeatureFlags.SPARSE_FUSED_FP32,
                        ]:
                            print(f"Running m = {m} n = {n} density = {density} storage = {compression_strategy}")
                            # Generate test case
                            x_fp32 = generate_x_fp32(n)
                            spqr_module, spqr_module_device = create_random(m, n, density, compression_strategy)

                            x_fp16_device = x_fp32.cuda(device=device).half()

                            deq_w = spqr_module.dequantize().to(device)

                            y_true = torch_mul(deq_w, x_fp16_device)
                            y = torch.zeros(m, dtype=torch.half, device=device)

                            _spqr_mul_fused(spqr_module_device, x_fp16_device, y, flag)

                            passed = torch.equal(y, y_true)

                            self.assertTrue(
                                passed,
                                msg=f"Failed for m = {m} n = {n} density = {density} compression_strategy = {compression_strategy}\ny={y}\ny_true={y_true}",
                            )


if __name__ == "__main__":
    unittest.main()
