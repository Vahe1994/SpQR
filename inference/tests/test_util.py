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

from inference import *


def generate_x_fp32(n, upper_bound=3):
    x_fp32 = ((torch.rand(n) - 0.5) * 4 * upper_bound).int()
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


def create_random(m, n, density, device: torch.device, bits=3, beta1=16, beta2=16,
                  sparse_compression_strategy=0) -> Tuple[SPQRModule, SPQRModule]:
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

    spqr_host = SPQRUncompressed(
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
        sparse_compression_strategy=sparse_compression_strategy)

    spqr_module = SPQRModule(spqr_host)
    spqr_module_device = SPQRModule(spqr_host)
    spqr_module_device.to_device(device)

    return spqr_module, spqr_module_device


def host_to_device(spqr_host: SPQRUncompressed, device: torch.device):
    row_offsets = spqr_host.row_offsets.cuda(device=device).int()
    col_ptr = spqr_host.col_ids.cuda(device=device).short()
    values = spqr_host.values.cuda(device=device).half()

    first_order = (
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
        first_order
    )

    return SPQRModule(
        spqr_host,
        spqr_host.m,
        spqr_host.n,
        spqr_host.bits,
        spqr_host.beta1,
        spqr_host.beta2,
        first_order.cuda(device=device),
        row_offsets.cuda(device=device).int(),
        col_ptr.cuda(device=device).short(),
        values.cuda(device=device).half(),
        torch.empty(spqr_host.nnz)
    )


def create_twos(m, n) -> Tuple[SPQRUncompressed, SPQRModule]:
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

    spqr_host = SPQRUncompressed(
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


def create_ones_random_2nd_order(m, n, density, device: torch.device, bits=3, beta1=16, beta2=16) -> Tuple[
    SPQRModule, SPQRModule]:
    W = torch.ones(m * n).char()

    num_first_order_groups = updiv(n, beta2) * m
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.ones(num_first_order_groups).char()
    W_z_raw = torch.zeros(num_first_order_groups).char()

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

    spqr_host = SPQRUncompressed(
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


def create_ones_random_1st_order(m, n) -> Tuple[SPQRUncompressed, SPQRModule]:
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

    spqr_host = SPQRUncompressed(
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


def create_ones_random_w(m, n) -> Tuple[SPQRUncompressed, SPQRModule]:
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

    spqr_host = SPQRUncompressed(
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


def create_ones(m, n, device: torch.device, bits=3, beta1=16, beta2=16, compression_strategy=0) -> Tuple[SPQRModule, SPQRModule]:
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

    spqr_host = SPQRUncompressed(
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
        sparse_compression_strategy=compression_strategy)

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

    spqr_host = SPQRUncompressed(
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


def create_random_from_sparse(m, n, row_offsets, col_ids, values, device: torch.device) -> Tuple[
    SPQRModule, SPQRModule]:
    spqr_host, spqr_device = create_random(m, n, device, device)

    spqr_host.row_offsets = row_offsets
    spqr_host.values = values
    spqr_host.col_ids = col_ids

    spqr_device.row_offsets = row_offsets.cuda(device=device)
    spqr_device.values = values.half().cuda(device=device)
    spqr_device.col_ids = col_ids.cuda(device=device)

    return spqr_host, spqr_device



def create_random_from_sparse_repeat(m, n, row_offsets, col_ids, values, rep, device: torch.device) -> Tuple[
    SPQRModule, SPQRModule]:
    if rep == 1:
        return create_random_from_sparse(m, n, row_offsets, col_ids, values, device)

    spqr_host, spqr_device = create_random(m * rep, n * rep, density=0, device=device)

    dense = torch.sparse_csr_tensor(row_offsets, col_ids, values, (m, n)).to_dense()

    tiled_dense = dense.tile((rep, rep))

    tiled_sparse_csr = tiled_dense.to_sparse_csr()

    spqr_host.row_offsets = tiled_sparse_csr.crow_indices().int()
    spqr_host.col_ids = tiled_sparse_csr.col_indices().short()
    spqr_host.values = random_like(tiled_sparse_csr.values())

    spqr_device.row_offsets = spqr_host.row_offsets.cuda(device=device)
    spqr_device.values = spqr_host.values.half().cuda(device=device)
    spqr_device.col_ids = spqr_host.col_ids.cuda(device=device)

    return spqr_host, spqr_device
