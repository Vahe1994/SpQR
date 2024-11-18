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
from dataclasses import dataclass
from typing import Tuple

import torch

from spqr import updiv, SparseStorageConfiguration, QuantizedLinear, SPQRLegacy, ModelArgs


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

    return r.crow_indices().int(), \
        r.values().half(), \
        r.col_indices().short(), \
        r._nnz()


@dataclass
class DenseWeightInitStrategy:
    randomize: bool = False
    set_all: torch.float16 = None
    arange: bool = False


@dataclass
class SparseWeightInitStrategy:
    sparsity: float = 0.


@dataclass
class MatrixBuilder:
    m: int
    n: int
    weights: DenseWeightInitStrategy
    first_order: DenseWeightInitStrategy
    second_order: DenseWeightInitStrategy


def create_spqr_quantized_matrix(m: int,
                                 n: int,
                                 weight_init_strategy: int = None,
                                 first_order_init_strategy: int = None,
                                 second_order_init_strategy: torch.float16 = None,
                                 density: float = 0.,
                                 sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR,
                                 in_perm=None) -> Tuple[QuantizedLinear, QuantizedLinear]:
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
        m,
        n,
        3,
        W,
        16,
        16,
        W_s,
        W_z,
        W_s_s,
        W_s_z,
        W_z_s,
        W_z_z,
        row_offsets,
        col_ids,
        values,
        in_perm
    )

    mod = QuantizedLinear.from_legacy(spqr_legacy, ModelArgs(3, 16, 16, sparse_storage), 'cpu')
    mod_device = QuantizedLinear.from_legacy(spqr_legacy, ModelArgs(3, 16, 16, sparse_storage), 'cuda')

    return mod, mod_device


def create_ones(m, n, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, 1, 1, 1, 0., sparse_storage, None)


def create_random(m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, None, None, None, density, sparse_storage, None)


def create_random_weights_ones(m, n, density,
                               sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, 1, None, None, density, sparse_storage, None)


def create_random_first_order_ones(m, n, density,
                                   sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, None, 1, None, density, sparse_storage, None)


def create_random_second_order_ones(m, n, density,
                                    sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, None, None, 1, density, sparse_storage, None)


def create_just_sparse(m, n, density, sparse_storage: SparseStorageConfiguration = SparseStorageConfiguration.CSR):
    return create_spqr_quantized_matrix(m, n, 0, 0, 0, density, sparse_storage, None)
