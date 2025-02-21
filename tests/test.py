import unittest
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from spqr_quant import QuantizedLinear
from spqr_quant.inference import FeatureFlags, ModelArgs, SparseStorageConfiguration, SPQRLegacy, updiv
from spqr_quant.inference_kernels.kernel_selector import (
    get_spqr_mul,
    get_spqr_mul_batched,
    get_spqr_mul_fused,
    get_torch_mul_timer,
)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def generate_x_fp32(n, upper_bound=2):
    x_fp32 = ((torch.rand(n) - 0.5) * 2 * upper_bound).int()
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


class TestSparseFp16BatchedRandomColumnMajor(unittest.TestCase):
    def test_sparse_random(self):
        print("")
        # Call this once just to trigger the annoying torch sparse warning.
        device = torch.device("cuda:0")
        for m in [32]:
            for k in [256, 512, 1024, 1 << 15]:
                for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:  # , 2, 4, 8]:
                    for density in [0.0, 0.1, 0.2]:
                        for compression_strategy in [
                            SparseStorageConfiguration.CSR,
                            SparseStorageConfiguration.PTCSR
                        ]:
                            for generator_strategy in [
                                "random"
                            ]:
                                for flag in [
                                    FeatureFlags.SPARSE_FUSED_FP32_COLUMN_MAJOR,
                                ]:
                                    print(
                                        f"Running m = {m} n = {n} k = {k} density = {density} storage = {compression_strategy} generator = {generator_strategy}"
                                    )

                                    # Generate test case
                                    x_fp32 = generate_x_fp32(k * n)
                                    if generator_strategy == "ones":
                                        x_fp32 = x_fp32 * 0 + 1
                                        spqr_module, spqr_module_device = create_ones(m, k)
                                    else:
                                        spqr_module, spqr_module_device = create_random(
                                            m, k, density, compression_strategy
                                        )
                                    x_fp16_device = x_fp32.cuda(device=device).half().reshape((1, n, k)).contiguous()

                                    deq_w = spqr_module.dequantize().to(device)
                                    linear = torch.nn.Linear(deq_w.shape[1], deq_w.shape[0], bias=False, device='cuda',
                                                             dtype=torch.half)
                                    with torch.no_grad():
                                        linear.weight = torch.nn.Parameter(deq_w, requires_grad=False)

                                    y = spqr_module_device.forward(x_fp16_device)
                                    y_true = linear.forward(x_fp16_device)

                                    passed = torch.equal(y, y_true)

                                    if not passed:
                                        print(y)
                                        print(y_true)

                                    self.assertTrue(
                                        passed,
                                        msg=f"Failed for m = {m} n = {n} k = {k} density = {density} compression_strategy = {compression_strategy}\ny={y}\ny_true={y_true}",
                                    )


if __name__ == "__main__":
    unittest.main()
