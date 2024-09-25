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

import numpy as np

import os

import torch
import spqr_cuda

from torch import Tensor as T, nn

import inference


# Utility functions

def list_flatten(W):
    if torch.is_tensor(W):
        return W.flatten()
    else:
        return torch.cat(W).flatten()


def updiv(x, y): return (x + y - 1) // y


# Data structures
class SPQRUncompressed:
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

        self.buff0 = allocate_compressed_buffers(m, n, beta1, beta2, 'cpu')
        spqr_cuda.tensor_compress_interleaved(m, n, bits, W, beta1, beta2, W_s, W_z, W_s_s, W_s_z, W_z_s, W_z_z, self.buff0)

class SPQRModule(torch.nn.Module):
    def __init__(self, spqr_host: SPQRUncompressed):
        super().__init__()
        self.m = spqr_host.m
        self.n = spqr_host.n
        self.bits = spqr_host.bits
        self.beta1 = spqr_host.beta1
        self.beta2 = spqr_host.beta2
        self.buff0 = spqr_host.buff0
        self.row_offsets = spqr_host.row_offsets
        self.col_vals = spqr_host.col_vals
        self.deq_w = None
        self.in_perm = spqr_host.in_perm
        self.out_perm = spqr_host.out_perm

        self.y = torch.zeros((1, 10, self.m), dtype=torch.float16, device=self.buff0.device)
        self.y_single = torch.zeros((1, 1, self.m), dtype=torch.float16, device=self.buff0.device)
        self._y = torch.zeros(self.m, dtype=torch.float16, device=self.buff0.device)

    def allocate_output_buffers(self):
        self.y = torch.zeros((1, 10, self.m), dtype=torch.float16, device=self.buff0.device)
        self._y = torch.zeros(self.m, dtype=torch.float16, device=self.buff0.device)
        self.y_single = torch.zeros((1, 1, self.m), dtype=torch.float16, device=self.buff0.device)

    def to_device(self, device: torch.device):
        self.buff0 = self.buff0.to(device=device)
        self.row_offsets = self.row_offsets.to(device=device)
        self.col_vals = self.col_vals.to(device=device)

    @property
    def nnz(self):
        return self.col_vals.shape[0]

    def _apply(self, fn):
        # Apply the function to all parameters and buffers
        super(SPQRModule, self)._apply(fn)

        # Apply the function to custom attributes
        self.buff0 = fn(self.buff0)
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

        if inner_dim == 10:
            for i in range(inner_dim):
                _x = x[..., i, :].flatten()
                if self.in_perm is not None:
                    _x = _x[self.in_perm]
                spqr_cuda.spqr_mul(
                    self.m,
                    self.n,
                    self.bits,
                    self.beta1,
                    self.beta2,
                    self.buff0,
                    self.row_offsets,
                    self.col_vals,
                    self.nnz,
                    # TODO: Might case a CPU regression
                    _x,
                    self._y,
                    FeatureFlag.SPARSE_FUSED_FP32_ASYNC)
                if self.out_perm is not None:
                    out_perm_long = self.out_perm
                    self._y = self._y[out_perm_long]
                self.y[0, i, :] = self._y
            return self.y[:, :inner_dim, :]
        else:
            _x = x
            # if self.in_perm is not None:
            #     _x = _x[self.in_perm]
            spqr_cuda.spqr_mul(
                self.m,
                self.n,
                self.bits,
                self.beta1,
                self.beta2,
                self.buff0,
                self.row_offsets,
                self.col_vals,
                self.nnz,
                # TODO: Might case a CPU regression
                _x,
                self.y_single,
                FeatureFlag.SPARSE_FUSED_FP32_ASYNC)
            return self.y_single


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
        p.row_offsets,
        p.col_vals,
        nnz,
        deq_w)
    return deq_w


def spqr_dequantize_dense_compressed(p: SPQRModule):
    return _spqr_dequantize(p, 0)


def spqr_dequantize_compressed(p: SPQRModule):
    return _spqr_dequantize(p, p.col_vals.shape[0])


def spqr_dequantize_dense(p: SPQRUncompressed):
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


def spqr_dequantize(p: SPQRUncompressed):
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


def spqr_mul_host(p: SPQRUncompressed, x, y_gt, y, dequantized_w):
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
    SPARSE_FUSED_FP32_ASYNC = SPARSE_FUSED_ALGORITHM | FP32 | IS_ASYNC
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
        spqr_device.row_offsets,
        spqr_device.col_vals,
        spqr_device.nnz,
        x,
        y,
        int(feature_flag)
    )


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
            spqr_device.row_offsets,
            spqr_device.col_vals,
            spqr_device.nnz,
            x,
            y,
            runs[i],
            feature_flag)

    return y, runs


def write_tensor(spqr_module: SPQRModule, path: str):
    torch.save(spqr_module, path)


def load_compressed_tensor(p: str) -> SPQRModule:
    spqr_module = torch.load(p)
    return spqr_module


def calculate_buffer_sizes(m, n, beta1, beta2):
    total_blocks = updiv(m, beta1) * updiv(n, beta2)
    per_tile_row = 1
    block_size = beta1 * per_tile_row
    return block_size * total_blocks


def allocate_compressed_buffers(m, n, beta1, beta2, device) -> torch.Tensor:
    sz0 = calculate_buffer_sizes(m, n, beta1, beta2)
    return torch.zeros(sz0, dtype=torch.int64, device=device)


def num_tiles(m, n, beta1, beta2):
    return (m // beta1) * (n // beta2)


def num_first_order_grups(m, n, beta1, beta2):
    return num_tiles(m, n, beta1, beta2) * beta1


def generate_random_nbit_tensor(sz, bits=3):
    d = (torch.rand(sz) * (1 << bits)).char()
    return d
