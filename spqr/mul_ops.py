import os
import torch

from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spqr')
CUDA_KERNEL = load(
    name="spqr_torch_lib",
    sources=[os.path.join(CUDA_FOLDER, "spqr_torch_lib.cpp"),
             os.path.join(CUDA_FOLDER, "spqr_cuda_kernel.cu")],
    extra_cuda_cflags=['-O3', '-arch=native']
)

torch.library.define(
    "spqr_torch_lib::spqr_mul", "(int m, int n, int bits, int beta1, int beta2, Tensor buff0, Tensor row_offsets, Tensor col_vals, int nnz, Tensor x, int f, Tensor Y, Tensor(Y!) out) -> ()"
)

torch.library.impl("spqr_torch_lib::spqr_mul", "default", CUDA_KERNEL.spqr_mul)


@torch.library.register_fake("spqr_torch_lib::spqr_mul")
def spqr_mul_meta(m, n, bits, beta1, beta2, buff0, row_offsets, col_vals, nnz, x, f, Y, out):
    """
    There are no allocation performed by this kernel.
    TODO: Add input sanitation.
    """
    return


def call_spqr_mul(*args):
    return torch.ops.spqr_torch_lib.spqr_mul(*args)


