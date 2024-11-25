import os

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
SPQR_CUDA = load(
    name="spqr_cuda",
    sources=[os.path.join(CUDA_FOLDER, "spqr_cuda.cpp"), os.path.join(CUDA_FOLDER, "spqr_cuda_kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-arch=native", "-lineinfo"],
)

torch.library.define(
    "spqr_cuda::dequantize_compressed",
    "(int m, int n, int bits, int beta1, int beta2, Tensor dense_weights, Tensor row_offsets, Tensor col_vals, int nnz, Tensor deq_w_tensor) -> ()",
)
torch.library.define(
    "spqr_cuda::spqr_mul",
    "(int m, int n, int bits, int beta1, int beta2, Tensor dense_weights, Tensor row_offsets, Tensor col_vals, int nnz, Tensor x, int f, Tensor Y, Tensor(Y!) out) -> ()",
)
torch.library.define(
    "spqr_cuda::spqr_mul_timer",
    "(int m, int n, int bits, int beta1, int beta2, Tensor dense_weights, Tensor row_offsets, Tensor col_vals, int nnz, Tensor x, Tensor Y, Tensor measurements, int f) -> ()",
)
torch.library.define(
    "spqr_cuda::torch_mul_timer",
    "(Tensor deq_w, Tensor x, Tensor y, Tensor measurements) -> ()",
)
torch.library.define(
    "spqr_cuda::tensor_compress_interleaved",
    "(int m, int n, int bits, Tensor W, int beta1, int beta2, Tensor W_s, Tensor W_z, Tensor W_s_s, Tensor W_s_z, Tensor W_z_s, Tensor W_z_z, Tensor row_offsets, Tensor row_offsets_output, Tensor col_vals, Tensor col_vals_interleaved, int sparse_strategy_compression, Tensor out) -> ()",
)
torch.library.define(
    "spqr_cuda::spqr_mul_fused",
    "(int m, int n, int bits, int beta1, int beta2, Tensor in_perm, Tensor dense_weights, Tensor row_offsets, Tensor col_vals, int nnz, Tensor x, int f, Tensor Y, Tensor(Y!) out) -> ()",
)

torch.library.impl("spqr_cuda::torch_mul_timer", "default", SPQR_CUDA.torch_mul_timer)
torch.library.impl("spqr_cuda::tensor_compress_interleaved", "default", SPQR_CUDA.tensor_compress_interleaved)
torch.library.impl("spqr_cuda::spqr_mul_timer", "default", SPQR_CUDA.spqr_mul_timer)
torch.library.impl("spqr_cuda::spqr_mul", "default", SPQR_CUDA.spqr_mul)
torch.library.impl("spqr_cuda::dequantize_compressed", "default", SPQR_CUDA.dequantize_compressed)
torch.library.impl("spqr_cuda::spqr_mul_fused", "default", SPQR_CUDA.spqr_mul_fused)


def call_spqr_mul(*args):
    return torch.ops.spqr_cuda.spqr_mul(*args)


def call_spqr_mul_timer(*args):
    return torch.ops.spqr_cuda.spqr_mul_timer(*args)


def call_torch_mul_timer(*args):
    return torch.ops.spqr_cuda.torch_mul_timer(*args)


def call_tensor_compress_interleaved(*args):
    return torch.ops.spqr_cuda.tensor_compress_interleaved(*args)


def call_dequantize_compressed(*args):
    return torch.ops.spqr_cuda.dequantize_compressed(*args)


def call_spqr_mul_fused(*args):
    return torch.ops.spqr_cuda.spqr_mul_fused(*args)


@torch.library.register_fake("spqr_cuda::spqr_mul")
def spqr_mul_meta(m, n, bits, beta1, beta2, dense_weights, row_offsets, col_vals, nnz, x, f, Y, out):
    return


@torch.library.register_fake("spqr_cuda::spqr_mul_timer")
def spqr_mul_timer_meta(m, n, bits, beta1, beta2, dense_weights, row_offsets, col_vals, nnz, x, f, Y, out):
    return


@torch.library.register_fake("spqr_cuda::spqr_mul_fused")
def spqr_mul_fused_meta(m, n, bits, beta1, beta2, in_perm, dense_weights, row_offsets, col_vals, nnz, x, f, Y, out):
    return
