import os
import time

import torch
import torch.nn as nn
from tqdm import trange

from datautils import get_loaders
from modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
)
from spqr_engine import Quantizer, SPQRUtil, quantize

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


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
    groupsize = groupsize or float("inf")
    qq_groupsize = qq_groupsize or float("inf")

    if groupsize is None:
        wbits_avg = wbits
    elif round_zero:
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
        wbits_avg += 32 * global_ol_n_share

    return round(wbits_avg, 2)


def quantize_model(model, args, device):
    """main entry point to functions for model quantization"""
    tick = time.time()
    if args.wbits == 16:
        print("not quantizing the model with args.wbits=16", flush=True)
        results = None, args.wbits
    elif args.nearest:
        results = quantize_nearest(model, args, device)
    else:
        print("Loading data ...")
        dataloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
        )
        results = quantize_spqr(model, dataloader, args, device)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
    ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def quantize_spqr(model, dataloader, args, device):
    print("\nStarting SPQR quantization ...")

    inps, forward_args = get_inps(model, dataloader, args, dev="cpu" if args.offload_activations else device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    quantizers = {}

    normal_outlier_count_global, w_count_global = 0, 0

    layers = get_layers(model)
    for i in range(len(layers)):
        print(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device  # quantized layer will return there
        print(f"{layer_dev_original=}")
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)  # noqa: F821

                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))
            for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                outs[j] = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
                if args.offload_activations:
                    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()

            torch.cuda.empty_cache()

            for sublayer_name in subset:
                print(f"Quantizing module {sublayer_name} of layer {i}")
                quantized = spqr_handlers[sublayer_name].quantize(
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
                    save_quantization=save,
                )

                if save:
                    quantized.save_quant_dict["sublayer_name"] = sublayer_name
                    full_path = save + "/" + str(i) + "/"
                    os.makedirs(full_path, exist_ok=True)
                    torch.save(quantized.save_quant_dict, full_path + sublayer_name)

                spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[sublayer_name].layer.weight.data.dtype
                )
                quantizers["model.layers.%d.%s" % (i, sublayer_name)] = ()  # to be updated

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()
                stats_payload[f"n_{sublayer_name}_ol_share"] = (normal_outliers_count / quantized.weight.numel()).item()
                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()

        out_losses = []
        for j in trange(args.nsamples, desc="calc outs after quantization", leave=False):
            outs_batch = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_dev))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                out_losses.append(outs_batch_loss.item())
            outs[j] = outs_batch
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        del outs_batch

        layers[i] = layer.to(layer_dev_original)
        del layer
        del spqr_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["ol_share"] = normal_outlier_count / max(w_count, 1)
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        stats_payload["Step"] = i

        normal_outlier_count_global += normal_outlier_count
        w_count_global += w_count

        print(stats_payload)

    print("=====================\nFinal stats:")
    print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")

    wbits_avg = get_average_number_of_bits(
        wbits=args.wbits,
        qq_scale_bits=args.qq_scale_bits,
        qq_zero_bits=args.qq_zero_bits,
        qqq_scale_bits=16,
        qqq_zero_bits=16,
        groupsize=args.groupsize,
        qq_groupsize=args.qq_groupsize,
        round_zero=args.round_zero,
        global_ol_n_share=normal_outlier_count_global / w_count_global,
    )
    if save:
        torch.save(vars(args), save + "/args.pt")
        already_saved_weights = set()
        for name, layer in nn.ModuleList(get_layers(model)).named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                already_saved_weights.add(layer.weight)
        not_quantized_weights = {
            name: param for name, param in model.named_parameters() if param not in already_saved_weights
        }
        torch.save(not_quantized_weights, save + "/not_quantized_weights.pt")

    if args.wandb:
        wandb.log({"outlier_share": normal_outlier_count_global / w_count_global})
        wandb.log({"wbits_avg": wbits_avg})
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})

    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers, wbits_avg


@torch.no_grad()
def quantize_nearest(model, args, dev):
    """Round-to-nearest quantization"""
    layers = get_layers(model)
    for i in trange(len(layers), desc="quantizing layers to nearest"):
        layer_dev = next(layers[i].parameters()).device
        layer = layers[i].to(dev)
        subset = find_sublayers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                next(iter(layer.parameters())).dtype
            )
        layers[i] = layer.to(layer_dev)
        del layer
        torch.cuda.empty_cache()
    return None, args.wbits


@torch.no_grad()
def perplexity_eval(model, testenc, args, dev):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(
        model, testenc, args, dev="cpu" if args.offload_activations else dev, nsamples=nsamples
    )
    outs = torch.zeros_like(inps)
    for k, v in forward_args.items():
        forward_args[k] = v.to(dev) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].to(dev).unsqueeze(0), **forward_args)[0]
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(dev)
    testenc = testenc.to(dev)

    nlls = []
    for i in range(nsamples):
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="none",
        help="Dataset name [c4, pajama, refinedweb, none, etc.] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified. Deprecated",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
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
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
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
        "--qq_zero_sym",
        action="store_true",
        help="enable sym=True in meta-quantization for groupwise zero, specifically",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=16,
        help="Quantize quantization scale in groups of this many scales",
    )
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
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    args = parser.parse_args()

    if args.dataset == "custom":
        print(
            "WARNING: `--custom_data_path` argument and `--dataset=custom` option are DEPRECATED. ",
            "Pass dataset path directly to `dataset` argument or use 'pajama', 'refinedweb'",
            "See README.md for examples.",
        )
        args.dataset = args.custom_data_path

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "SpQR_run")
            + f"_wbits_{args.wbits}"
            + f"_groupsize_{args.groupsize}"
            + f"_qq_scale_bits_{args.qq_scale_bits}"
            + f"_qq_zero_bits_{args.qq_zero_bits}"
            + f"_qq_groupsize_{args.qq_groupsize}"
            + f"_outl_{args.outlier_threshold}"
            + f"_permord_{args.permutation_order}"
            + f"{'_new_eval' if args.new_eval else ''}"
        )
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("============  Loading model... ============")
    model = get_model(args.model_path, args.load, args.dtype).train(False)

    print("\n============ Quantizing model... ============")
    if args.wbits < 16 and args.load:
        print("\n Warning: You are quantizing quantized model!")
    quantize_model(model, args, device)

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
            eval_mode=True,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args, device)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
