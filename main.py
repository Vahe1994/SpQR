import os
import time

import wandb
import torch
import torch.nn as nn
from tqdm import trange

from datautils import get_loaders
from spqr_engine import SPQRUtil, Quantizer, quantize


def get_llama(model_path):
    import torch

    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # preserving
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path, 
        local_files_only=True, 
        torch_dtype="auto"
    )
    model.seqlen = 2048
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
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
    # if not quantized stats are in full precision
    qq_scale_bits = qq_scale_bits or 16
    qq_zero_bits = qq_zero_bits or 16
    groupsize = groupsize or float('inf')
    qq_groupsize = qq_groupsize or float('inf')

    if round_zero:
        wbits_avg = wbits + (qq_scale_bits + wbits) / groupsize + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
    else:
        wbits_avg = wbits + (qq_scale_bits + qq_zero_bits) / groupsize +  2 * (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)

    # correct accounting for outliers
    if global_ol_n_share > 0:
        wbits_avg += 32 * global_ol_n_share

    return round(wbits_avg, 2)


@torch.no_grad()
def llama_sequential(model, dataloader, args, dev):
    print("\nStarting SPQR compression ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

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

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    quantizers = {}
    normal_outlier_count_global, w_count_global = 0, 0

    for i in range(len(layers)):
        print(f"\n------------------------------------------------------------------\nStarting layer {i}")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}

        start_time = time.time()
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Quantizing module {name} of layer {i}")
                quantized = gptq[name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                )

                gptq[name].layer.weight.data = quantized.weight.to(gptq[name].layer.weight.data.dtype)
                quantizers["model.layers.%d.%s" % (i, name)] = () # to be updated

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()

                stats_payload[f"n_{name}_ol_share"] = round((normal_outliers_count / quantized.weight.numel()).item(), 6)

                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()

        # upload inputs back to the device
        if args.offload_activations:
            inps = inps.to(device)
            outs = outs.to(device)        

        if not args.skip_out_loss:
            outs_tmp = outs.clone()

        for j in trange(args.nsamples, desc="applying", leave=False):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        if args.skip_out_loss:
             out_losses = torch.full((1,), torch.nan)
        else:
            out_losses = (outs - outs_tmp).float().square().view(
                outs.shape[0], -1
            ).mean(dim=1).sqrt() / outs.view(outs.shape[0], -1).float().std(dim=1)
            del outs_tmp
            
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["ol_share"] = round(normal_outlier_count / w_count, 6)
        stats_payload["out_loss"] = torch.mean(out_losses).item()
        stats_payload["Step"] = i

        normal_outlier_count_global += normal_outlier_count
        w_count_global += w_count

        print(stats_payload)

    print("=====================\nFinal stats:")
    print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")

    wbits_avg = get_average_number_of_bits(
        args.wbits,
        args.qq_scale_bits,
        args.qq_zero_bits,
        16,
        16,
        args.groupsize,
        args.qq_groupsize,
        args.round_zero,
        normal_outlier_count_global / w_count_global
    )

    if args.wandb:
        wandb.log({"outlier_share": normal_outlier_count_global / w_count_global})
        wandb.log({"wbits_avg": wbits_avg})

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, args, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i, end=", ", flush=True)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                    next(iter(layer.parameters())).dtype
                )

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\nperplexity = {ppl.item():.4f}")

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("model_path", type=str, help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["custom", "wikitext2", "ptb", "c4"],
        default="none",
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--load_from_saved",
        type=str,
        default=None,
        help="Path to load if specified.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument("--perchannel", action="store_true", help="fit a unique quantizer to each output dim")
    parser.add_argument(
        "--qq_scale_bits",
        type=int,
        default=None,
        help="Quantize quantization scale with this many bits (default=do not quantize)",
    )
    parser.add_argument(
        "--round_zero",
        type=int,
        default=None,
        help='whether to allow non-integer "zero" when quantizing weights non-symmetrically',
    )
    parser.add_argument(
        "--qq_zero_bits",
        type=int,
        default=None,
        help='Quantize quantization "zero" with this many bits (default=do not quantize)',
    )
    parser.add_argument(
        "--qq_zero_sym", action="store_true", help="enable sym=True in meta-quantization for groupwise zero, specifically"
    )
    parser.add_argument("--qq_groupsize", type=int, default=16, help="Quantize quantization scale in groups of this many scales")

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="relative threshold for     outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity",
    )

    parser.add_argument("--save", type=str, default="", help="Save quantized checkpoint under this name.")
    parser.add_argument(
        "--save_safetensors", type=str, default="", help="Save quantized `.safetensors` checkpoint under this name."
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument("--benchmark", type=int, default=0, help="Number of tokens to use for benchmarking.")
    parser.add_argument(
        "--check", action="store_true", help="Whether to compute perplexity during benchmarking for verification."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to use wandb or store locally."
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="",
        help="Directory where to store local wandb files.",
    )
    parser.add_argument(
        "--wandb_exp_name",
        type=str,
        default="SpQR",
        help="Suffix of wandb experiments name.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        raise NotImplementedError()
    else:
        model = get_llama(args.model_path).train(False)

    if args.load_from_saved:
        dataloader = torch.load(args.load_from_saved)[: args.nsamples]
        testloader = None
    else:
        assert args.dataset != "custom"
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model_path=args.model_path, seqlen=model.seqlen
        )

    if args.wandb:
        args.exp_name = (
            args.wandb_exp_name
            + "_wbits_"
            + str(args.wbits)
            + "_groupsize_"
            + str(args.groupsize)
            + "_qq_scale_bits_"
            + str(args.qq_scale_bits)
            + "_qq_zero_bits_"
            + str(args.qq_zero_bits)
            + "_qq_groupsize_"
            + str(args.qq_groupsize)
            + "_outl_"
            + str(args.outlier_threshold)
            + "_permord_"
            + str(args.permutation_order)
        )
        neweval_str = ""
        if args.new_eval:
            neweval_str = "_new_eval"
        wandb.init(
            name=args.exp_name,
            dir=args.wandb_dir,
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, args, device)
        print(time.time() - tick)

    if args.benchmark:
        raise NotImplementedError()

    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model_path=args.model_path, seqlen=model.seqlen)
        print(dataset)
        args.dataset_name = dataset
        llama_eval(model, testloader, args, device)

    if args.save or args.save_safetensors:
        raise NotImplementedError()
