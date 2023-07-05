from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def safediv(x: Tensor, value: int):
    return x.div(value, rounding_mode='floor')


def pack8to4(x: Tensor):
    return torch.stack([safediv(x, 16), x % 16], dim=-1).flatten(-2, -1)


def pack4to8(x: Tensor):
    return 16 * x[..., ::2] + x[..., 1::2]


class QParameter(nn.Module):

    def __init__(self, qweight: Tensor, scale: Tensor, zero: Tensor, outliers = None, bits: int = 8) -> None:
        """
        qweight: 
            quantized weight, stored in 8 bits
        scale:
            quantization scales
        zero:
        outliers:
            sparse tensor with outliers
        bits:
            storage
        """
        assert bits in [4, 8]
        super().__init__()
        self.bits = bits
        # infer groupsize from qweight and stats
        self.groupsize = None if scale.shape[1] == 1 else qweight.shape[1] // scale.shape[1]
        # data
        self.register_buffer('qweight', self._pack(qweight, bits))
        self.register_buffer('scale', scale)
        self.register_buffer('zero', zero)
        if outliers is not None:
            self.outliers = nn.Parameter(outliers)
        else:
            self.outliers = None

    def _repeat_if_needed(self, x: Tensor):
        if self.groupsize:
            return x.repeat_interleave(self.groupsize, dim=1)
        return x
    
    @staticmethod
    def _pack(x: Tensor, bits: int):
        return pack4to8(x) if bits == 4 else x
    
    @staticmethod
    def _unpack(x: Tensor, bits: int):
        return pack8to4(x) if bits == 4 else x

    def dequantize(self):
        qweight = self._unpack(self.qweight, self.bits).to(self.scale.dtype)
        scale = self._repeat_if_needed(self.scale)
        zero = self._repeat_if_needed(self.zero).to(self.scale.dtype)
        weight = scale * (qweight - zero)
        if self.outliers is not None:
            weight.add_(self.outliers)
        return weight


class QLinear(nn.Module):

    def __init__(self, qparam: QParameter, bias: Optional[Tensor] = None):
        super().__init__()
        self.qparam = qparam
        self.bias = bias

    def forward(self, x):
        weight = self.qparam.dequantize() 
        output = F.linear(x, weight, self.bias)
        return output
