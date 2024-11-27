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
import os
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
import torch
from torch import Tensor as T, nn

from .inference_kernels.cuda_kernel import (
    call_dequantize_compressed,
    call_spqr_mul,
    call_spqr_mul_fused,
    call_tensor_compress_interleaved,
)
from .sparse_util import init_ptcsr, merge_col_val


# Utility functions
class SparseStorageConfiguration(StrEnum):
    CSR = "csr"
    PTCSR = "ptcsr"
    OPTIMIZE_LATENCY = "optimize_latency"


@dataclass
class SPQRLegacy:
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
    col_ids: T
    values: T
    in_perm: T


@dataclass
class ModelArgs:
    bits: int
    beta1: int
    beta2: int
    sparse_compression: SparseStorageConfiguration

    @staticmethod
    def from_file(legacy_model_path: str, compression_strategy: str):
        b = torch.load(os.path.join(legacy_model_path, "args.pt"))
        bits = b["wbits"]
        beta1 = b["qq_groupsize"]
        beta2 = b["groupsize"]
        sparse_compression = SparseStorageConfiguration(compression_strategy)
        return ModelArgs(bits=bits, beta1=beta1, beta2=beta2, sparse_compression=sparse_compression)


class QuantizedLinear(torch.nn.Module):
    def __init__(self, rows, cols, bits, beta1, beta2, dense_weights, row_offsets, col_vals, in_perm):
        """
        This class stores the fully compressed bits=3-bit beta2=16 x beta1=16 SpQR mxn tensor.

        @param rows: Matrix height.
        @param cols: Matrix width.
        @param bits: Size of the weights and first order data quantization.
        @param beta1: Tile width.
        @param beta2: Tile height (see SpQR publication for more details).
        @param dense_weights: The weights, first and second order data is stored in dense_weights. A single row tile
        looks is stored in a 64-bit chunk as follows:

                   |  w_s  | w_z | w_0 | ... | w_15 | c_0 | unused
        bits(64):      3      3     3           3      8      2

        w_s - first order quantized row scale
        w_z - first order quantized row zero-point.

        w_i - i-th quantized weight

        c_0 - partial 8-bit chunk of the second order which the kernel recovers via warp shuffle reduction opeartion.

        @param row_offsets: int32 CSR Row offsets if row_offsets.shape[0] == m + 1. PTCSR row offsets otherwise.
        @param col_vals: int32 | values | column id | buffer.
                                 16-bit    16-bit
                                 higher     lower
        @param in_perm: Optional input permutation (see SpQR publication for more details).
        """
        super().__init__()
        self.m = rows
        self.n = cols
        self.bits = bits
        self.beta1 = beta1
        self.beta2 = beta2

        self.dense_weights = nn.Parameter(dense_weights, requires_grad=False)
        self.row_offsets = nn.Parameter(row_offsets, requires_grad=False)
        self.col_vals = nn.Parameter(col_vals, requires_grad=False)

        if in_perm is None:
            self.in_perm = None
        else:
            if in_perm.dtype != torch.int32:
                raise ValueError(f"Invalid dtype={in_perm.dtype} for in_perm passed, torch.uint32 expected")
            self.in_perm = nn.Parameter(in_perm, requires_grad=False)

    @staticmethod
    def create_placehodler(
        rows,
        cols,
        bits,
        beta1,
        beta2,
        dense_weights_shape: int,
        row_offsets_shape: int,
        col_vals_shape: int,
        in_perm_shape: int,
    ):
        dense_weights = nn.Parameter(torch.empty(dense_weights_shape, dtype=torch.int64), requires_grad=False)
        row_offsets = nn.Parameter(torch.empty(row_offsets_shape, dtype=torch.int32), requires_grad=False)
        col_vals = nn.Parameter(torch.empty(col_vals_shape, dtype=torch.int32), requires_grad=False)
        in_perm = nn.Parameter(torch.empty(in_perm_shape, dtype=torch.int32), requires_grad=False)

        return QuantizedLinear(rows, cols, bits, beta1, beta2, dense_weights, row_offsets, col_vals, in_perm)

    @staticmethod
    def _calculate_weight_buffer_size(m, n, beta1, beta2):
        total_blocks = updiv(m, beta1) * updiv(n, beta2)
        per_tile_row = 1
        block_size = beta1 * per_tile_row
        return block_size * total_blocks

    @staticmethod
    def _allocate_weight_buffer(m, n, beta1, beta2, device) -> torch.Tensor:
        sz0 = QuantizedLinear._calculate_weight_buffer_size(m, n, beta1, beta2)
        return torch.zeros(sz0, dtype=torch.int64, device=device)

    @staticmethod
    def from_legacy(spqr_legacy: SPQRLegacy, model_args: ModelArgs, device):
        """
        Converts the int8 legacy format storage into it's compressed counterpart.
        @param spqr_legacy: SPQRLegacy object.
        @param model_args: Global model parameters.
        @param device: Device.
        @return: Fully compressed SpQR tensor.
        """
        dense_weights = QuantizedLinear._allocate_weight_buffer(
            spqr_legacy.m, spqr_legacy.n, model_args.beta1, model_args.beta2, "cpu"
        )

        col_vals = merge_col_val(spqr_legacy.col_ids, spqr_legacy.values)
        row_offsets_output = spqr_legacy.row_offsets

        if model_args.sparse_compression == SparseStorageConfiguration.PTCSR:
            row_offsets_output, col_vals_output = init_ptcsr(row_offsets_output)
        else:
            col_vals_output = col_vals

        call_tensor_compress_interleaved(
            spqr_legacy.m,
            spqr_legacy.n,
            model_args.bits,
            spqr_legacy.W,
            model_args.beta1,
            model_args.beta2,
            spqr_legacy.W_s,
            spqr_legacy.W_z,
            spqr_legacy.W_s_s,
            spqr_legacy.W_s_z,
            spqr_legacy.W_z_s,
            spqr_legacy.W_z_z,
            spqr_legacy.row_offsets,
            row_offsets_output,
            col_vals,
            col_vals_output,
            0 if model_args.sparse_compression == SparseStorageConfiguration.CSR else 1,
            dense_weights,
        )

        def pack_uint16_to_uint32(tensor: torch.Tensor) -> torch.Tensor:
            """
            Packs a uint16 PyTorch tensor into a uint32 tensor by combining
            two consecutive uint16 values into a single uint32 value.

            Args:
                tensor (torch.Tensor): Input uint16 tensor with an even number of elements.

            Returns:
                torch.Tensor: A uint32 tensor half the size of the input.
            """
            if tensor.dtype != torch.uint16:
                raise TypeError("Input tensor must be of dtype torch.uint16.")
            if tensor.numel() % 2 != 0:
                raise ValueError("Tensor length must be even to pair elements.")

            # Convert to NumPy for bitwise operations
            numpy_array = tensor.numpy().view(np.uint16)
            packed_array = (numpy_array[1::2].astype(np.uint32) << 16) | numpy_array[0::2]

            # Convert back to PyTorch
            return torch.from_numpy(packed_array).int()

        mod = QuantizedLinear(
            spqr_legacy.m,
            spqr_legacy.n,
            model_args.bits,
            model_args.beta1,
            model_args.beta2,
            dense_weights,
            row_offsets_output,
            col_vals_output,
            pack_uint16_to_uint32(spqr_legacy.in_perm.to(dtype=torch.uint16))
            if spqr_legacy.in_perm is not None
            else None,
        )

        return mod.to(device=device)

    def _spqr_dequantize(self, nnz):
        """
        Internal method, see dequantize and dequantize_dense_only for reference.
        """
        deq_w = torch.zeros(self.m, self.n).half().contiguous()
        call_dequantize_compressed(
            self.m,
            self.n,
            self.bits,
            self.beta1,
            self.beta2,
            self.dense_weights,
            self.row_offsets,
            self.col_vals,
            nnz,
            deq_w,
        )
        return deq_w

    def dequantize_dense_only(self):
        """
        Dequantize the matrix using only the dense weight data.

        Possibly useful during debugging, testing or benchmarking.

        @return: Dense-only dequantized SpQR matrix.
        """
        return self._spqr_dequantize(0)

    def dequantize(self):
        """
        @return: Dequantized SpQR matrix.
        """
        return self._spqr_dequantize(self.col_vals.shape[0])

    @property
    def nnz(self):
        """
        @return: Number of non-zeros of the sparse matrix, optionally including zero-padding in the case of PTCSR.
        """
        return self.col_vals.shape[0]

    @property
    def density(self) -> float:
        """
        @return: Sparse matrix density.
        """
        return self.col_vals.shape[0] / (self.m * self.n)

    @property
    def sparsity(self) -> float:
        """
        @return: Sparsity.
        """
        return 1 - self.density

    def should_reorder(self) -> bool:
        """
        @return: Input reordering is an optional argument. Check if we should reorder before matvec.
        """
        return self.in_perm is not None and torch.numel(self.in_perm) != 0

    @torch.no_grad()
    def forward(self, x: T) -> T:
        """
        Forward matmul operation. The kernel currently only supports matvec. Therefore, to fully implement matmuls,
        a for loop is used which is horribly inefficient, but will do for now.
        @param x: Input tensor.
        @return: A tensor resulting from a multiplication between the SpQR tensor and input tensor x.
        """
        inner_dim = x.shape[1]

        if inner_dim == 1:
            y = torch.empty(self.m, dtype=torch.float16, device=self.dense_weights.device)
        else:
            y = torch.empty((1, inner_dim, self.m), dtype=torch.float16, device=self.dense_weights.device)

        for i in range(inner_dim):
            if inner_dim != 1:
                _x = x[..., i, :].flatten()
                _y = y[0, i]
            else:
                _x = x.view(-1)
                _y = y
            if self.should_reorder():
                call_spqr_mul_fused(
                    self.m,
                    self.n,
                    self.bits,
                    self.beta1,
                    self.beta2,
                    self.in_perm,
                    self.dense_weights,
                    self.row_offsets,
                    self.col_vals,
                    self.nnz,
                    _x,
                    int(FeatureFlags.SPARSE_FUSED_FP32),
                    _y,
                    _y,
                )
            else:
                call_spqr_mul(
                    self.m,
                    self.n,
                    self.bits,
                    self.beta1,
                    self.beta2,
                    self.dense_weights,
                    self.row_offsets,
                    self.col_vals,
                    self.nnz,
                    _x,
                    int(FeatureFlags.SPARSE_FUSED_FP32),
                    _y,
                    _y,
                )

        return y.reshape((1, inner_dim, self.m))


def updiv(x, y):
    """
    @return: Utility method: updivision between x and y.
    """
    return (x + y - 1) // y


FP16 = 0b0
FP32 = 0b1

FULL = 0b0 << 1  # Dense and sparse mul
DENSE_ONLY = 0b1 << 1  #

TORCH = 0b1 << 3
IS_ASYNC = 0b1 << 4
SPARSE_FUSED_ALGORITHM = FULL | (0b1 << 8)


class FeatureFlags(IntEnum):
    SPARSE_FUSED_FP32 = SPARSE_FUSED_ALGORITHM | FP32
    SPARSE_FUSED_FP32_ASYNC = SPARSE_FUSED_ALGORITHM | FP32 | IS_ASYNC
    TORCH_FP16 = TORCH | FP16

    def pretty(self):
        if self.value == FeatureFlags.TORCH_FP16:
            return "Torch FP16"
        elif self.value == FeatureFlags.SPARSE_FUSED_FP32:
            return "Sparse Fused FP32"
        else:
            raise "Prettify not found for value {self.value}"
