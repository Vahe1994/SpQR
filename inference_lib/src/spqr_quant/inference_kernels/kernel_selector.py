import torch


def get_spqr_mul_fused():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.spqr_cuda.spqr_mul_fused


def get_spqr_mul_timer():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.spqr_cuda.spqr_mul_timer


def get_torch_mul_timer():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.spqr_cuda.torch_mul_timer


def get_spqr_mul():
    from .cuda_kernel import CUDA_FOLDER

    return torch.ops.spqr_cuda.spqr_mul
