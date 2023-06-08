import torch
import torch.nn as nn

from .spqr.spqr import SPQRUtil
from .spqr.quant_groups import Quantizer, quantize

def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_average_number_of_bits(
    wbits: int = 3,
    qq_scale_bits: int = 3,
    qq_zero_bits: int = 3,
    qqq_scale_bits: int = 16,
    qqq_zero_bits: int = 16,
    groupsize: int = 16,
    qq_groupsize: int = 16,
    round_zero: bool = False,
    global_ol_n_share: float = 0.00,
):
    if groupsize is None:
        wbits_avg = wbits

    else:
        qq_scale_bits = qq_scale_bits or 16
        qq_zero_bits = qq_zero_bits or 16

        if round_zero:
            wbits_avg = (
                wbits + (qq_scale_bits + wbits) / groupsize + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
            )
        else:
            wbits_avg = (
                wbits
                + (qq_scale_bits + qq_zero_bits) / groupsize
                + 2 * (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
            )

        # correct accounting for outliers
        if global_ol_n_share > 0:
            wbits_avg += (32 - wbits) * global_ol_n_share

    return round(wbits_avg, 2)


@torch.no_grad()
def falcon_sequential(model, dataloader, args, dev):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    alibi = cache['alibi']

    quantizers = {}
    normal_outlier_count_global, column_and_row_outlier_count_global, w_count_global = (
        0,
        0,
        0,
    )

    for i in range(len(layers)):
        print(f"\n------------------------------------------------------------------\nStarting layer {i}")
        normal_outlier_count, column_and_row_outlier_count, w_count = 0, 0, 0
        layer = layers[i]
        full = find_layers(layer)

        # put inps to layer device
        layer_device = next(layer.parameters()).device
        #
        if inps.device != layer_device:
            inps = inps.to(layer_device)

        if args.true_sequential:
            sequential = [
                ["self_attention.query_key_value"],
                ["self_attention.dense"],
                ["mlp.dense_h_to_4h"],
                ["mlp.dense_4h_to_h"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            if args.nearest:
                for name in subset:
                    quantizer = Quantizer()
                    quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                        next(iter(layer.parameters())).dtype
                    )
                continue

            gptq = {}
            for name in subset:
                gptq[name] = SPQRUtil(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            for h in handles:
                h.remove()

            if args.offload_activations:
                inps = inps.cpu()
                outs = outs.cpu()
                torch.cuda.empty_cache()

            for name in subset:
                print(f"Quantizing module {name} of layer {i}")
                quantized = gptq[name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    mse=args.mse,
                    qq_groupsize=args.qq_groupsize,
                    qq_mse=args.qq_mse,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    outlier_cols_enable=args.outlier_cols_enable,
                    outlier_rows_enable=args.outlier_rows_enable,
                    outlier_percentile_base=args.outlier_percentile_base,
                    outlier_percentile_multiple=args.outlier_percentile_multiple,
                    simplified_outliers=args.simplified_outliers,
                )

                gptq[name].layer.weight.data = quantized.weight.to(gptq[name].layer.weight.data.dtype)
                quantizers["model.layers.%d.%s" % (i, name)] = ()  # TODO

                # OUTLIER STATS per module:
                om2 = quantized.unstructured_outlier_mask.clone()  # special matrix to calculate number of individual outliers
                om2[:, quantized.outlier_column_indices] = 0
                om2[quantized.outlier_row_indices, :] = 0
                normal_outliers_count = om2.to(torch.int32).sum()

                normal_outlier_count += normal_outliers_count.item()
                column_and_row_outlier_count += (
                    len(quantized.outlier_row_indices) * quantized.weight.shape[1]
                    + len(quantized.outlier_column_indices) * quantized.weight.shape[0]
                    - len(quantized.outlier_row_indices) * len(quantized.outlier_column_indices)
                )
                w_count += quantized.weight.numel()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer
        if not args.nearest:
            del gptq
        del layer
        torch.cuda.empty_cache()

        # upload inputs back to the device
        if args.offload_activations:
            inps = inps.to(layer_device)
            outs = outs.to(layer_device)

        inps, outs = outs, inps

        normal_outlier_count_global += normal_outlier_count
        column_and_row_outlier_count_global += column_and_row_outlier_count
        w_count_global += w_count

    wbits_avg = get_average_number_of_bits(
        args.wbits,
        args.qq_scale_bits,
        args.qq_zero_bits,
        16,
        16,
        args.groupsize,
        args.qq_groupsize,
        args.round_zero,
        normal_outlier_count_global / max(w_count_global, 1),
    )

    model.config.use_cache = use_cache
    return wbits_avg
