import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quant_groups import dequantize


__all__ = ["QLinear"]


def pack4to8(x: Tensor):
    return 16 * x[..., :x.shape[-1] // 2] + x[..., x.shape[-1] // 2:]


def pack8to4(x: Tensor):
    return torch.cat([x // 16, x % 16], dim=-1)


class QLinear(nn.Module):

    def __init__(
        self, 
        quant_weights,
        scale,
        zero,
        scale_qq_scale=[],
        scale_qq_zero=[],
        zero_qq_scale=[],
        zero_qq_zero=[],
        outliers_matrix=None,
        bias=None,
        perm=None,
        bits=8
    ) -> None:
        assert bits in [4, 8]
        super().__init__()
        self.bits = bits
        self.in_features = quant_weights.shape[1]
        self.out_features = quant_weights.shape[0]
        self.perm = perm
        if perm is not None:
            self.invperm = perm.argsort()
        else:
            self.invperm = None

        if bits == 4:
            quant_weights = pack4to8(quant_weights)
        self.register_buffer('quant_weights', quant_weights)

        if len(scale_qq_scale) > 0:
            self.register_buffer('scale', torch.stack(scale, dim=-1))
            self.scale_qq_scale = nn.Parameter(torch.stack(scale_qq_scale, dim=-1))
            self.scale_qq_zero = nn.Parameter(torch.stack(scale_qq_zero, dim=-1))
        else:
            self.scale = nn.Parameter(torch.cat(scale, dim=-1))
            self.scale_qq_scale = None
            self.scale_qq_zero = None

        if len(zero_qq_scale) > 0:
            self.register_buffer('zero', torch.stack(zero, dim=-1))
            self.zero_qq_scale = nn.Parameter(torch.stack(zero_qq_scale, dim=-1))
            self.zero_qq_zero = nn.Parameter(torch.stack(zero_qq_zero, dim=-1))
        else:
            self.zero = nn.Parameter(torch.cat(zero, dim=-1))
            self.zero_qq_scale = None
            self.zero_qq_zero = None

        if outliers_matrix is not None:
            self.register_buffer('outlier_ids', outliers_matrix.indices())
            self.outlier_vals = nn.Parameter(outliers_matrix.values())
        else:
            self.outlier_ids = None
            self.outlier_vals = None  

        if bias is not None:
            if perm is not None:
                # reorder elements
                bias = bias[perm]
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None) 

        self.gradient_checkpointing = False
        
    def get_outliers(self):
        if self.outlier_ids is not None:
            return torch.sparse_coo_tensor(
                self.outlier_ids, self.outlier_vals.float(), (self.out_features, self.in_features)
            )
        return None

    def get_weight_without_outliers(self):
        quant_weights = self.quant_weights
        if self.bits == 4:
            quant_weights = pack8to4(quant_weights)

        if self.scale_qq_scale is not None:
            scale = dequantize(self.scale, self.scale_qq_scale, self.scale_qq_zero).view(quant_weights.shape[0], -1, 1)
        else:
            scale = self.scale.view(quant_weights.shape[0], -1, 1)

        if self.zero_qq_scale is not None:
            zero = dequantize(self.zero, self.zero_qq_scale, self.zero_qq_zero).view(quant_weights.shape[0], -1, 1)
        else:
            zero = self.zero.view(quant_weights.shape[0], -1, 1)

        num_groups = scale.shape[1]
        weight = dequantize(quant_weights.view(quant_weights.shape[0], num_groups, -1), scale, zero).view_as(quant_weights)
        # zero weights on outlier positions
        if self.outlier_ids is not None:
            weight[self.outlier_ids[0], self.outlier_ids[1]] = 0
        return weight
        
    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype
        if self.perm is not None:
            input = input[..., self.perm]
        input = input.float()
        # get weight without outliers
        weight = self.get_weight_without_outliers()
        # get outliers
        outliers = self.get_outliers()
        out = F.linear(input, weight, self.bias)
        if outliers is not None:
            out = out.float()
            with torch.cuda.amp.autocast(enabled=False):
                out.add_(torch.sparse.mm(
                    outliers, input.to(torch.float32).view(-1, self.in_features).T
                ).T.view(*input.shape[:-1], self.out_features))
            out = out.to(input_dtype)
        if self.perm is not None:
            input = input[..., self.invperm]
        return out
        