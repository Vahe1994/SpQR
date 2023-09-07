from __future__ import annotations

import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import tqdm

from quant_groups import Quantizer, dequantize, quantize
from weight_permutation import get_permutation_order


class SPQRUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def quantize(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        save_quantization: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        save_quant_dict = {}
        perm = get_permutation_order(self.H, weight, permutation_order)

        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                weight.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                weight.device
            )  # shape = [out_features, in_features]

        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        H_inv_cho_diag = torch.diag(H_inv_cho)
        del H

        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # prepare outlier detection
        outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
        del H_inv

        outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
        in_group_index = -1  # index of current group of input features, for group quantizer purposes

        quantization_errors = torch.zeros_like(weight)
        unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

        block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
        block_start_iter = tqdm(block_start_iter, leave=False) if verbose else block_start_iter
        for block_start in block_start_iter:
            block_end = min(block_start + blocksize, in_dim)
            for column_index in range(block_start, block_end):
                if column_index % groupsize == 0:
                    # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                    in_group_index += 1
                    group_weight = weight[:, column_index : column_index + groupsize]

                    if simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                        quantizer.find_params(group_weight, weight=True)

                    else:
                        # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                        # step 1: fit quantizer on a leave-one-out version of weights, i.e. in each group, drop one weight at a time
                        assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                        group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                        loo_quantization_error_sq = get_leave_one_out_error(
                            group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                        )
                        # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                        likely_unstructured_outlier_mask = (
                            loo_quantization_error_sq > unstructured_outlier_threshold
                        ).float()

                        non_outlier_mask = 1 - likely_unstructured_outlier_mask
                        mean_over_non_outliers = torch.sum(
                            group_weight * non_outlier_mask, dim=1, keepdim=True
                        ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                        group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                            1 - non_outlier_mask
                        )
                        quantizer.find_params(group_weight_without_outliers, weight=True)
                        del group_diag_hessian_inv_cho, loo_quantization_error_sq
                        del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                    if save_quantization:
                        if quantizer.qq_scale_bits is not None:
                            save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
                            save_quant_dict["quant_layer_scale_qq_scale"].append(
                                quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_scale_qq_zero"].append(
                                quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_scale"].append(
                                quantizer.scale.to(save_quant_dict["save_float_dtype"])
                            )

                        if quantizer.qq_zero_bits is not None and (
                            (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
                        ):
                            save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
                            save_quant_dict["quant_layer_zero_qq_scale"].append(
                                quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_zero_qq_zero"].append(
                                quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_zeros"].append(
                                quantizer.zero.to(save_quant_dict["save_float_dtype"])
                            )
                    del group_weight

                weight_quant_i = quantize(
                    weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                )
                weight_i_quantized = dequantize(weight_quant_i, quantizer.scale, quantizer.zero).reshape_as(
                    weight[:, column_index]
                )

                delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                quantization_errors[:, column_index] = (
                    delta_weight_i / H_inv_cho[column_index, column_index]
                )  # [out_dim]

                if unstructured_outlier_threshold != float("inf"):
                    unstructured_outlier_mask[:, column_index] = (
                        quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                    )
                    # re-quantize without outliers
                    is_outlier = unstructured_outlier_mask[:, column_index].float()

                    weight_quant_i = quantize(
                        (weight[:, column_index] * (1 - is_outlier)).unsqueeze(1),
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )
                    weight_i_quantized_wo_outliers = dequantize(
                        weight_quant_i, quantizer.scale, quantizer.zero
                    ).reshape_as(weight[:, column_index])
                    weight_i_quantized = (
                        weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                    )  # [out_dim]

                    if save_quantization:
                        save_quant_dict["outliers_matrix"][:, column_index] = weight[:, column_index] * is_outlier

                    del weight_i_quantized_wo_outliers

                    delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                    quantization_errors[:, column_index] = (
                        delta_weight_i / H_inv_cho[column_index, column_index]
                    )  # [out_dim]

                if save_quantization:
                    save_quant_dict["quant_weights"].append(weight_quant_i.to(torch.int8))

                weight[:, column_index] = weight_i_quantized
                weight[:, column_index + 1 : block_end].addr_(
                    quantization_errors[:, column_index],
                    H_inv_cho[column_index, column_index + 1 : block_end],
                    alpha=-1,
                )

            weight[:, block_end:].addmm_(
                quantization_errors[:, block_start:block_end],
                H_inv_cho[block_start:block_end, block_end:],
                alpha=-1,
            )

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            weight = weight[:, invperm]

        if save_quantization:
            save_quant_dict["perm"] = perm.to(torch.int32)
            save_quant_dict["keep_last_columns"] = 0
            save_quant_dict["blocksize"] = 128
            save_quant_dict["weight_shape"] = weight.shape
            save_quant_dict["groupsize"] = groupsize if groupsize else weight.shape[1]
            save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
            save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()

        return QuantizationResult(
            weight=weight,
            perm=perm,
            quantization_errors=quantization_errors,
            unstructured_outlier_threshold=unstructured_outlier_threshold,
            unstructured_outlier_mask=unstructured_outlier_mask,
            save_quant_dict=save_quant_dict,
        )


class QuantizationResult(NamedTuple):
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""

    weight: torch.FloatTensor  # dequantized(quantized(weight)), same shape as the original
    perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

    quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    save_quant_dict: dict


def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize_dequantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = (
        ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    )
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize_dequantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error
