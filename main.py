import time

import wandb
from tqdm import trange

from spqr_engine import SPQRUtil, Quantizer, quantize
from modelutils import *

from datautils import get_loaders


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

    if round_zero:
        wbits_avg = (
            wbits
            + (qq_scale_bits + wbits) / groupsize
            + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
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


def compress_model(model, dataloader, args, dev):
    """main entry point to functions for model compression"""
    tick = time.time()
    if args.load:
        raise NotImplementedError()
    elif args.wbits == 16:
        print("not compressing the model with args.wbits=16", flush=True)
        pass
    elif args.nearest:
        model = compress_nearest(model, args, dev)
    else:
        model = compress_spqr(model, dataloader, args, dev)
    print(f"compression time: {time.time() - tick:.1f}")
    return model


@torch.no_grad()
def get_inps(model, args, data_iterable, device, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    model.get_input_embeddings().to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )

    forward_arg_names = (
        ("attention_mask",)
        if "llama" in args.model_path
        else ("attention_mask", "alibi")
    )
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
                model(batch[0].to(device))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(device))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.get_input_embeddings().to(torch.device("cpu"))
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    model.config.use_cache = use_cache
    return inps, forward_args


@torch.no_grad()
def compress_spqr(model, dataloader, args, dev):
    inps, forward_args = get_inps(model, args, dataloader, dev)
    outs = torch.zeros_like(inps)

    print("\nStarting SPQR compression ...")
    use_cache = model.config.use_cache  # TODO find proper context for no cache use
    model.config.use_cache = False

    quantizers = {}
    normal_outlier_count_global, w_count_global = 0, 0

    layers = get_layers(model)
    for i in range(len(layers)):
        print(f"\n---------------- Layer {i} ----------------")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}

        start_time = time.time()
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            spqr_handlers = {}
            for name in subset:
                spqr_handlers[name] = SPQRUtil(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in trange(
                args.nsamples, desc="calc outs before compression", leave=False
            ):
                outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
            for h in handles:
                h.remove()

            if args.offload_activations:
                inps = inps.cpu()
                outs = outs.cpu()
                torch.cuda.empty_cache()

            for name in subset:
                print(f"Quantizing module {name} of layer {i}")
                quantized = spqr_handlers[name].quantize(
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

                spqr_handlers[name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[name].layer.weight.data.dtype
                )
                quantizers["model.layers.%d.%s" % (i, name)] = ()  # to be updated

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(
                    torch.int32
                ).sum()

                stats_payload[f"n_{name}_ol_share"] = round(
                    (normal_outliers_count / quantized.weight.numel()).item(), 6
                )

                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()

        # upload inputs back to the device
        if args.offload_activations:
            inps = inps.to(device)
            outs = outs.to(device)

        out_losses = []
        for j in trange(args.nsamples, desc="calc outs after compression", leave=False):
            outs_batch = layer(inps[j].unsqueeze(0), **forward_args)[0]
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j])
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= (
                    outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                )
                out_losses.append(outs_batch_loss.item())
            outs[j] = outs_batch
        del outs_batch

        layers[i] = layer.cpu()
        del layer
        del spqr_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["ol_share"] = round(normal_outlier_count / max(w_count, 1), 6)
        stats_payload["out_loss"] = torch.mean(out_losses).item()
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

    if args.wandb:
        wandb.log({"outlier_share": normal_outlier_count_global / w_count_global})
        wandb.log({"wbits_avg": wbits_avg})

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def compress_nearest(model, args, dev):
    """Round-to-nearest compression"""
    layers = get_layers(model)
    for i in trange(len(layers), desc="compressing layers to nearest"):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def perplexity_eval(model, testenc, args, dev):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(model, args, testenc, device=dev, nsamples=nsamples)
    outs = torch.zeros_like(inps)

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    move_head(model, dev)
    testenc = testenc.to(dev)

    nlls = []
    for i in range(nsamples):
        lm_logits = get_lm_logits(inps[i], model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    move_head(model, torch.device("cpu"))

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
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
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

    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    parser.add_argument(
        "--save_safetensors",
        type=str,
        default="",
        help="Save quantized `.safetensors` checkpoint under this name.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
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
        model = get_model(args.model_path, args.dtype).train(False)

    if args.load_from_saved:
        dataloader = torch.load(args.load_from_saved)[: args.nsamples]
        testloader = None
    else:
        assert args.dataset != "custom"
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
        )

    if args.wandb:
        args.exp_name = (
            args.wandb_exp_name
            + f"_wbits_{args.wbits}"
            + f"_groupsize_{args.groupsize}"
            + f"_qq_scale_bits_{args.qq_scale_bits}"
            + f"_qq_zero_bits_{args.qq_zero_bits}"
            + f"_qq_groupsize_{args.qq_groupsize}"
            + f"_outl_{args.outlier_threshold}"
            + f"_permord_{args.permutation_order}"
            + f"{'_new_eval' if args.new_eval else ''}"
        )
        wandb.init(  # TODO add args for entity and project name or describe usage of env variables
            name=args.exp_name,
            dir=args.wandb_dir,
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")
    else:
        wandb.init(mode="disabled")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = compress_model(model, dataloader, args, device)

    if args.benchmark:
        raise NotImplementedError()

    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model_path=args.model_path, seqlen=model.seqlen
        )
        print(dataset)
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args, device)

    if args.save or args.save_safetensors:
        raise NotImplementedError()
