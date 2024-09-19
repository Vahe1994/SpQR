#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum
from typing import Tuple

import numpy as np

import os

import torch
import spqr_cuda

from torch import Tensor as T, nn

import time

import warnings

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


# Utility functions

def list_flatten(W):
    if torch.is_tensor(W):
        return W.flatten()
    else:
        return torch.cat(W).flatten()


def updiv(x, y): return (x + y - 1) // y


# Data structures
class SPQRHost:
    m: int
    n: int
    bits: int
    W: T
    beta1: int
    beta2: int
    W_s: T
    W_z: T
    W_s_s: T
    W_s_z: T
    W_z_s: T
    W_z_z: T
    row_offsets: T
    col_vals: T
    dense_row_count: int
    row_ids: T
    buff0: T
    buff1: T

    @property
    def nnz(self) -> int:
        return self.col_vals.shape[0]

    def __init__(self, m, n, bits, W, beta1, beta2, W_s, W_z, W_s_s, W_s_z, W_z_s, W_z_z, row_offsets, col_ids, values,
                 in_perm=None, out_perm=None):
        self.m = m
        self.n = n
        self.bits = bits
        self.W = W
        self.beta1 = beta1
        self.beta2 = beta2
        self.W_s = W_s
        self.W_z = W_z
        self.W_s_s = W_s_s
        self.W_s_z = W_s_z
        self.W_z_s = W_z_s
        self.W_z_z = W_z_z
        self.row_offsets = row_offsets
        self.col_ids = col_ids
        self.values = values
        self.in_perm = in_perm
        self.out_perm = out_perm

        if self.values.shape[0] != 0:
            self.col_vals = values.view(torch.int16).to(torch.int64).bitwise_left_shift(16).bitwise_or(
                col_ids.view(torch.int16).to(torch.int64)).to(torch.int32)
        else:
            self.col_vals = torch.zeros(0)

        ROW_CUTOFF = 0.02

        nnzs = row_offsets.diff()
        nnz_ids = (nnzs * -1).sort().indices
        nnzs = nnzs[nnz_ids]
        self.row_ids = nnz_ids.short()
        row_densities = nnzs / n
        self.dense_row_count = (row_densities >= ROW_CUTOFF).sum().item()

        self.buff0, self.buff1 = allocate_compressed_buffers(m, n, beta1, beta2, 'cpu')

        spqr_cuda.tensor_compress_interleaved(m, n, bits, W, beta1, beta2, W_s, W_z, W_s_s, W_s_z, W_z_s, W_z_z,
                                              self.buff0, self.buff1)


class SPQRModule(nn.Module):
    def __init__(self, spqr_host: SPQRHost, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = spqr_host.m
        self.n = spqr_host.n
        self.bits = spqr_host.bits
        self.beta1 = spqr_host.beta1
        self.beta2 = spqr_host.beta2
        self.buff0 = spqr_host.buff0
        self.buff1 = spqr_host.buff1
        self.row_ids = spqr_host.row_ids
        self.row_offsets = spqr_host.row_offsets
        self.col_vals = spqr_host.col_vals
        self.dense_row_count = spqr_host.dense_row_count
        self.deq_w = None
        self.in_perm = spqr_host.in_perm
        self.out_perm = spqr_host.out_perm

    def to_device(self, device: torch.device):
        self.buff0 = self.buff0.to(device=device)
        self.buff1 = self.buff1.to(device=device)
        self.row_ids = self.row_ids.to(device=device)
        self.row_offsets = self.row_offsets.to(device=device)
        self.col_vals = self.col_vals.to(device=device)
        self.dense_row_count = self.dense_row_count

    @property
    def nnz(self):
        return self.col_vals.shape[0]

    def _apply(self, fn, **kwargs):
        # Apply the function to all parameters and buffers
        super(SPQRModule, self)._apply(fn)

        # Apply the function to custom attributes
        self.buff0 = fn(self.buff0)
        self.buff1 = fn(self.buff1)
        self.row_ids = fn(self.row_ids)
        self.row_offsets = fn(self.row_offsets)
        self.col_vals = fn(self.col_vals)

        if self.in_perm is not None:
            self.in_perm = fn(self.in_perm)

        if self.out_perm is not None:
            self.out_perm = fn(self.out_perm)

        return self

    @property
    def density(self) -> float:
        return self.col_vals.shape[0] / (self.m * self.n)

    @property
    def sparsity(self) -> float:
        return 1 - self.density

    def forward(self, x: T) -> T:
        inner_dim = x.shape[1]
        y = torch.zeros((1, inner_dim, self.m), dtype=x.dtype, device=x.device)

        # start_time = time.time()
        for i in range(inner_dim):
            _y = torch.zeros(self.m, dtype=x.dtype, device=x.device)
            _x = x[0, i, :].flatten()
            if self.in_perm is not None:
                _x = _x[self.in_perm]
            spqr_cuda.spqr_mul(
                self.m,
                self.n,
                self.bits,
                self.beta1,
                self.beta2,
                self.buff0,
                self.buff1,
                self.row_ids,
                self.row_offsets,
                self.col_vals,
                self.nnz,
                self.dense_row_count,
                # TODO: Might case a CPU regression
                _x,
                _y,
                FeatureFlag.SPARSE_FUSED_FP32)
            if self.out_perm is not None:
                out_perm_long = self.out_perm
                _y = _y[out_perm_long]
            y[0, i, :] = _y
        # end_time = time.time()
        # duration = end_time - start_time
        # print(f'\t\t{duration:.10f}')
        return y


# Compression

def compress(W, bit_count):
    W = list_flatten(W)
    value_count = W.shape[0]

    # Values per bucket
    values_per_bucket = 64 // bit_count

    total_buckets = (value_count + values_per_bucket - 1) // values_per_bucket
    out = torch.zeros((total_buckets), dtype=torch.int64)

    # Position within bucket
    buff_id = 0

    # Current bucket
    bucket_id = 0

    # Current bucket
    bucket = np.uint64(0)

    for i in range(value_count):
        bucket |= np.uint64(W[i]) << np.uint64(buff_id * bit_count)
        buff_id += 1

        if buff_id == values_per_bucket:
            out[bucket_id] = bucket
            buff_id = 0
            bucket = np.uint64(0)
            bucket_id += 1

    if buff_id > 0:
        out[bucket_id] = bucket

    return out


def _spqr_dequantize(p: SPQRModule, nnz):
    deq_w = torch.zeros(p.m, p.n).half().contiguous()
    spqr_cuda.spqr_dequantize_compressed(
        p.m,
        p.n,
        p.bits,
        p.beta1,
        p.beta2,
        p.buff0,
        p.buff1,
        p.row_offsets,
        p.col_vals,
        nnz,
        deq_w)
    return deq_w


def spqr_dequantize_dense_compressed(p: SPQRModule):
    return _spqr_dequantize(p, 0)


def spqr_dequantize_compressed(p: SPQRModule):
    return _spqr_dequantize(p, p.col_vals.shape[0])


def spqr_dequantize_dense(p: SPQRHost):
    deq_w = torch.zeros(p.m, p.n).float().contiguous()
    spqr_cuda.spqr_dequantize_host(
        p.m,
        p.n,
        p.bits,
        p.W,
        p.beta1,
        p.beta2,
        p.W_s,
        p.W_z,
        p.W_s_s,
        p.W_s_z,
        p.W_z_s,
        p.W_z_z,
        p.values,
        p.row_offsets,
        p.col_ids,
        0,  # This makes sure that we only quantize the dense part
        deq_w)
    return deq_w


def spqr_dequantize_dense(p: SPQRHost):
    deq_w = torch.zeros(p.m, p.n).float().contiguous()
    spqr_cuda.spqr_dequantize_host(
        p.m,
        p.n,
        p.bits,
        p.W,
        p.beta1,
        p.beta2,
        p.W_s,
        p.W_z,
        p.W_s_s,
        p.W_s_z,
        p.W_z_s,
        p.W_z_z,
        p.values,
        p.row_offsets,
        p.col_ids,
        0,  # This makes sure that we only quantize the dense part
        deq_w)
    return deq_w


def spqr_dequantize(p: SPQRHost):
    deq_w = torch.zeros(p.m, p.n).float().contiguous()
    spqr_cuda.spqr_dequantize_host(
        p.m,
        p.n,
        p.bits,
        p.W,
        p.beta1,
        p.beta2,
        p.W_s,
        p.W_z,
        p.W_s_s.float(),
        p.W_s_z.float(),
        p.W_z_s.float(),
        p.W_z_z.float(),
        p.values.float(),
        p.row_offsets,
        p.col_ids,
        p.nnz,
        deq_w
    )
    return deq_w


def spqr_mul_host(p: SPQRHost, x, y_gt, y, dequantized_w):
    spqr_cuda.spqr_mul_host_fp32(
        p.m,
        p.n,
        p.bits,
        p.W,
        p.beta1,
        p.beta2,
        p.W_s,
        p.W_z,
        p.W_s_s,
        p.W_s_z,
        p.W_z_s,
        p.W_z_z,
        p.values,
        p.row_offsets,
        p.col_ids,
        p.nnz,
        x,
        y_gt,
        y,
        dequantized_w)


def empty_sparse_tile():
    return 0, 0, torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()


def empty_sparse_csr():
    return torch.Tensor(), torch.Tensor(), torch.Tensor(), 0


FP16 = 0b0
FP32 = 0b1

FULL = 0b0 << 1  # Dense and sparse mul
DENSE_ONLY = 0b1 << 1  #

SPARSE_ALGORITHM0 = FULL | (0b0 << 2)
SPARSE_ALGORITHM_NAIVE = FULL | (0b1 << 2)

TORCH = 0b1 << 3

IS_ASYNC = 0b1 << 4

SPARSE_SHARED_ALGORITHM = FULL | (0b1 << 5)
SPARSE_SINGLE_ALGORITHM = FULL | (0b1 << 6)
SPARSE_CUSPARSE_ALGORITHM = FULL | (0b1 << 7)
SPARSE_FUSED_ALGORITHM = FULL | (0b1 << 8)
SPARSE_SHARED_BASELINE_ALGORITHM = FULL | (0b1 << 9)
SPARSE_SHARED_MIXTURE_ALGORITHM = FULL | (0b1 << 10)


class FeatureFlag(IntEnum):
    DENSE_ONLY_FP32 = DENSE_ONLY | FP32
    DENSE_ONLY_FP16 = DENSE_ONLY | FP16
    SPARSE_ALGO0_FP16 = SPARSE_ALGORITHM0 | FP16
    SPARSE_ALGO0_FP32 = SPARSE_ALGORITHM0 | FP32
    SPARSE_NAIVE_FP16 = SPARSE_ALGORITHM_NAIVE | FP16
    SPARSE_SHARED_FP16 = SPARSE_SHARED_ALGORITHM | FP16
    SPARSE_SINGLE_FP16 = SPARSE_SINGLE_ALGORITHM | FP16
    SPARSE_SHARED_BASELINE_FP16 = SPARSE_SHARED_BASELINE_ALGORITHM | FP16
    SPARSE_CUSPARSE_FP16 = SPARSE_CUSPARSE_ALGORITHM | FP16
    SPARSE_FUSED_FP16 = SPARSE_FUSED_ALGORITHM | FP16
    SPARSE_FUSED_FP32 = SPARSE_FUSED_ALGORITHM | FP32
    SPARSE_NAIVE_FP32 = SPARSE_ALGORITHM_NAIVE | FP32
    TORCH_FP16 = TORCH | FP16
    TORCH_FP32 = TORCH | FP32
    SPARSE_MIXTURE_FP16 = SPARSE_SHARED_MIXTURE_ALGORITHM | FP16
    SPARSE_MIXTURE_FP32 = SPARSE_SHARED_MIXTURE_ALGORITHM | FP32

    def pretty(self):
        if self.value == FeatureFlag.TORCH_FP16:
            return 'Torch FP16'
        elif self.value == FeatureFlag.SPARSE_MIXTURE_FP16:
            return 'SPQR Kernel'
        elif self.value == FeatureFlag.SPARSE_MIXTURE_FP32:
            return 'SPQR Kernel (FP32)'
        elif self.value == FeatureFlag.DENSE_ONLY_FP16:
            return 'SPQR Kernel Dense Only'
        elif self.value == FeatureFlag.DENSE_ONLY_FP32:
            return 'SPQR Kernel Dense Only (FP32)'
        elif self.value == FeatureFlag.SPARSE_SHARED_FP16:
            return 'Sparse With Sharing'
        elif self.value == FeatureFlag.SPARSE_SHARED_BASELINE_FP16:
            return 'Sparse Without Sharing'
        elif self.value == FeatureFlag.SPARSE_FUSED_FP16:
            return 'Sparse Fused FP16'
        elif self.value == FeatureFlag.SPARSE_FUSED_FP32:
            return 'Sparse Fused FP32'
        else:
            raise 'Pretty print not found for value {self.value}'


def spqr_mul(spqr_device: SPQRModule, x, y, feature_flag: FeatureFlag):
    spqr_cuda.spqr_mul(
        spqr_device.m,
        spqr_device.n,
        spqr_device.bits,
        spqr_device.beta1,
        spqr_device.beta2,
        spqr_device.buff0,
        spqr_device.buff1,
        spqr_device.row_ids,
        spqr_device.row_offsets,
        spqr_device.col_vals,
        spqr_device.nnz,
        spqr_device.dense_row_count,
        x,
        y,
        int(feature_flag)
    )


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


def spqr_mul_timer(spqr_device: SPQRModule, x, feature_flag: FeatureFlag, num_runs):
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
            spqr_device.buff1,
            spqr_device.row_ids,
            spqr_device.row_offsets,
            spqr_device.col_vals,
            spqr_device.nnz,
            spqr_device.dense_row_count,
            x,
            y,
            runs[i],
            feature_flag)

    return y, runs


# Factory

def calculate_buffer_sizes(m, n, beta1, beta2):
    total_blocks = updiv(m, beta1) * updiv(n, beta2)

    # For 32 bits, we need two slots per row
    # per_row = 2

    # 6 bits for s/z and 16 3-bits for weights per row
    per_row = 1

    block_size = beta1 * per_row

    return block_size * total_blocks, total_blocks


def allocate_compressed_buffers(m, n, beta1, beta2, device):
    sz0, sz1 = calculate_buffer_sizes(m, n, beta1, beta2)
    return (torch.zeros(sz0, dtype=torch.int64, device=device),
            torch.zeros(sz1, dtype=torch.int64, device=device))


def num_tiles(m, n, beta1, beta2):
    return (m // beta1) * (n // beta2)


def num_first_order_grups(m, n, beta1, beta2):
    return num_tiles(m, n, beta1, beta2) * beta1


def generate_random_nbit_tensor(sz, bits=3):
    d = (torch.rand(sz) * (1 << bits)).char()
    return d


def generate_x_fp32(n, upper_bound=3):
    x_fp32 = ((torch.rand(n) - 0.5) * 2 * upper_bound).int()
    return x_fp32.float()


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


def create_random(m, n, density, device: torch.device, bits=3, beta1=16, beta2=16) -> Tuple[SPQRModule, SPQRModule]:
    W_quantized = generate_3bit(m * n)
    W = W_quantized.char()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = generate_3bit(num_first_order_groups)
    W_z_raw = generate_3bit(num_first_order_groups)
    W_s = W_s_raw.char()
    W_z = W_z_raw.char()

    W_s_s = generate_x_fp32(num_second_order_groups).half()
    W_s_z = generate_x_fp32(num_second_order_groups).half()
    W_z_s = generate_x_fp32(num_second_order_groups).half()
    W_z_z = generate_x_fp32(num_second_order_groups).half()

    if density == 0:
        values = torch.zeros(0).half()
        row_offsets = torch.zeros(m + 1).int()
        col_ids = torch.zeros(0).short()
    else:
        row_offsets, values, col_ids, nnz = random_csr_host(m, n, density)

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values)

    spqr_module = SPQRModule(spqr_host)
    spqr_module_device = SPQRModule(spqr_host)
    spqr_module_device.to_device(device)

    return spqr_module, spqr_module_device


def host_to_device(spqr_host: SPQRHost, device: torch.device):
    row_offsets = spqr_host.row_offsets.cuda(device=device).int()
    col_ptr = spqr_host.col_ids.cuda(device=device).short()
    values = spqr_host.values.cuda(device=device).half()

    first_order, second_order = (
        allocate_compressed_buffers(spqr_host.m, spqr_host.n, spqr_host.beta1, spqr_host.beta2, 'cpu'))

    spqr_cuda.tensor_compress_interleaved(
        spqr_host.m,
        spqr_host.n,
        spqr_host.bits,
        spqr_host.W_dequantized.char(),
        spqr_host.beta1,
        spqr_host.beta2,
        spqr_host.W_s_raw,
        spqr_host.W_z_raw,
        spqr_host.W_s_s.half(),
        spqr_host.W_s_z.half(),
        spqr_host.W_z_s.half(),
        spqr_host.W_z_z.half(),
        first_order,
        second_order
    )

    return compress_sparse_device(SPQRModule(
        spqr_host.m,
        spqr_host.n,
        spqr_host.bits,
        spqr_host.beta1,
        spqr_host.beta2,
        first_order.cuda(device=device),
        second_order.cuda(device=device),
        row_offsets.cuda(device=device).int(),
        col_ptr.cuda(device=device).short(),
        values.cuda(device=device).half(),
        torch.empty(spqr_host.nnz)
    ))


def create_twos(m, n) -> Tuple[SPQRHost, SPQRModule]:
    beta1, beta2 = 16, 16
    bits = 3

    def c(E):
        return compress(E, bits)

    W_dequantized = torch.ones(m * n).float()
    W = c(W_dequantized.int()).int()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.ones(num_first_order_groups).char()
    W_z_raw = torch.zeros(num_first_order_groups).char()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.rand(num_second_order_groups).float()
    W_s_z = torch.zeros(num_second_order_groups).float()
    W_z_s = torch.rand(num_second_order_groups).float()
    W_z_z = torch.zeros(num_second_order_groups).float()

    values = torch.zeros(1).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(1)
    nnz = 0

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        W_dequantized=W_dequantized,
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)

    spqr_device = host_to_device(spqr_host)

    return spqr_host, spqr_device


def create_ones_random_2nd_order(m, n) -> Tuple[SPQRHost, SPQRModule]:
    beta1, beta2 = 16, 16
    bits = 3

    def c(E):
        return compress(E, bits)

    W_dequantized = torch.ones(m * n).float()
    W = c(W_dequantized.int()).int()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.ones(num_first_order_groups).char()
    W_z_raw = torch.zeros(num_first_order_groups).char()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.rand(num_second_order_groups).float()
    W_s_z = torch.rand(num_second_order_groups).float()
    W_z_s = torch.rand(num_second_order_groups).float()
    W_z_z = torch.rand(num_second_order_groups).float()

    values = torch.zeros(1).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(1)
    nnz = 0

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        W_dequantized=W_dequantized,
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)

    spqr_device = host_to_device(spqr_host)

    return spqr_host, spqr_device


def create_ones_random_1st_order(m, n) -> Tuple[SPQRHost, SPQRModule]:
    beta1, beta2 = 16, 16
    bits = 3

    def c(E):
        return compress(E, bits)

    W_dequantized = (torch.rand(m * n) * 8).int().float()
    W = c(W_dequantized.int()).int()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = (torch.rand(num_first_order_groups) * 8).char()
    W_z_raw = (torch.rand(num_first_order_groups) * 8).char()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.ones(num_second_order_groups).float()
    W_s_z = torch.zeros(num_second_order_groups).float()
    W_z_s = torch.ones(num_second_order_groups).float()
    W_z_z = torch.zeros(num_second_order_groups).float()

    values = torch.zeros(1).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(1)
    nnz = 0

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        W_dequantized=W_dequantized,
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)

    spqr_device = host_to_device(spqr_host)

    return spqr_host, spqr_device


def create_ones_random_w(m, n) -> Tuple[SPQRHost, SPQRModule]:
    beta1, beta2 = 16, 16
    bits = 3

    def c(E):
        return compress(E, bits)

    W_dequantized = (torch.rand(m * n) * 8).int().float()
    W = c(W_dequantized.int()).int()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.ones(num_first_order_groups).char()
    W_z_raw = torch.zeros(num_first_order_groups).char()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.ones(num_second_order_groups).float()
    W_s_z = torch.zeros(num_second_order_groups).float()
    W_z_s = torch.ones(num_second_order_groups).float()
    W_z_z = torch.zeros(num_second_order_groups).float()

    values = torch.zeros(1).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(1)
    nnz = 0

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        W_dequantized=W_dequantized,
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)

    spqr_device = host_to_device(spqr_host)

    return spqr_host, spqr_device


def random_csr_host(m, n, density):
    r = ((torch.rand(m, n) <= density) * (torch.ones(m, n) * 1).int()).to_sparse_csr()

    return r.crow_indices().int(), \
        r.values().half(), \
        r.col_indices().short(), \
        r._nnz()


def create_ones(m, n, device: torch.device, bits=3, beta1=16, beta2=16) -> Tuple[SPQRModule, SPQRModule]:
    W = torch.ones(m * n).char()

    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s = torch.ones(m * n).char()
    W_z = torch.zeros(m * n).char()

    W_s_s = torch.ones(num_second_order_groups).half()
    W_s_z = torch.zeros(num_second_order_groups).half()
    W_z_s = torch.ones(num_second_order_groups).half()
    W_z_z = torch.zeros(num_second_order_groups).half()

    values = torch.zeros(0).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(0)

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values)

    spqr_module = SPQRModule(spqr_host)
    spqr_module_device = SPQRModule(spqr_host)
    spqr_module_device.to_device(device)
    return spqr_module, spqr_module_device


def create_just_sparse(m, n, density):
    beta1, beta2 = 16, 16
    bits = 3

    def c(E):
        return compress(E, bits)

    W_dequantized = torch.zeros(m * n).float()
    W = c(W_dequantized.int()).int()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.zeros(num_first_order_groups).char()
    W_z_raw = torch.zeros(num_first_order_groups).char()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.zeros(num_second_order_groups).float()
    W_s_z = torch.zeros(num_second_order_groups).float()
    W_z_s = torch.zeros(num_second_order_groups).float()
    W_z_z = torch.zeros(num_second_order_groups).float()

    row_offsets, values, col_ids, nnz = random_csr_host(m, n, density)

    spqr_host = SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        W_dequantized=W_dequantized,
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)

    spqr_device = host_to_device(spqr_host)

    return spqr_host, spqr_device


def create_random_from_sparse(m, n, row_offsets, col_ids, values, device: torch.device) -> Tuple[SPQRHost, SPQRModule]:
    spqr_host, spqr_device = create_random(m, n, device)

    spqr_host.row_offsets = row_offsets
    spqr_host.values = values
    spqr_host.col_ids = col_ids

    spqr_device.row_offsets = row_offsets.cuda(device=device)
    spqr_device.values = values.half().cuda(device=device)
    spqr_device.col_ids = col_ids.cuda(device=device)

    return spqr_host, compress_sparse_device(spqr_device)


class ModelArgs:
    bits: int
    beta1: int
    beta2: int

    def __init__(self, model_path: str):
        b = torch.load(os.path.join(model_path, 'args.pt'))
        self.bits = b['wbits']
        self.beta1 = b['qq_groupsize']
        self.beta2 = b['groupsize']


class Model:
    def __init__(self, model_path: str):
        self.args = ModelArgs(model_path)


def write_tensor(spqr_host: SPQRHost, path: str):
    spqr_module = SPQRModule(spqr_host)

    torch.save(spqr_module, path)


def load_compressed_tensor(p: str) -> SPQRModule:
    spqr_module = torch.load(p)
    spqr_module.in_perm = spqr_module.perm.long()
    spqr_module.out_perm = None
    return spqr_module


def load_original_tensor(p: str, model_args: ModelArgs) -> SPQRHost:
    bits = model_args.bits
    beta1 = model_args.beta1
    beta2 = model_args.beta2

    t = torch.load(p, map_location='cpu')

    W = t['quant_weights']
    m = W.shape[0]
    n = W.shape[1]
    W = list_flatten(W)
    W_s = list_flatten(t['quant_layer_scale'])
    W_z = list_flatten(t['quant_layer_zeros'])

    perm = t['perm']

    outliers_matrix = t['outliers_matrix'].to_sparse_csr()

    col_ids = outliers_matrix.col_indices().short()
    values = outliers_matrix.values().half()

    return SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=list_flatten(W),
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=list_flatten(t['quant_layer_scale_qq_scale']),
        W_s_z=list_flatten(t['quant_layer_scale_qq_zero']),
        W_z_s=list_flatten(t['quant_layer_zero_qq_scale']),
        W_z_z=list_flatten(t['quant_layer_zero_qq_zero']),
        row_offsets=outliers_matrix.crow_indices().int(),
        col_ids=col_ids,
        values=values,
        in_perm=perm.long()
    )


def create_random_from_sparse_repeat(m, n, row_offsets, col_ids, values, rep, device: torch.device) -> Tuple[
    SPQRHost, SPQRModule]:
    if rep == 1:
        return create_random_from_sparse(m, n, row_offsets, col_ids, values, device)

    spqr_host, spqr_device = create_random(m * rep, n * rep, device)

    dense = torch.sparse_csr_tensor(row_offsets, col_ids, values, (m, n)).to_dense()

    tiled_dense = dense.tile((rep, rep))

    tiled_sparse_csr = tiled_dense.to_sparse_csr()

    spqr_host.row_offsets = tiled_sparse_csr.crow_indices().int()
    spqr_host.col_ids = tiled_sparse_csr.col_indices().short()
    spqr_host.values = random_like(tiled_sparse_csr.values())

    spqr_device.row_offsets = spqr_host.row_offsets.cuda(device=device)
    spqr_device.values = spqr_host.values.half().cuda(device=device)
    spqr_device.col_ids = spqr_host.col_ids.cuda(device=device)

    return spqr_host, compress_sparse_device(spqr_device)
