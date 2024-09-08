import os
import sys
from enum import Enum

import torch
import torch.nn as nn

from inference.module import LLama, Mode
from modelutils import get_lm_logits

torch.set_printoptions(sci_mode=False)

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False

@torch.no_grad()
def get_inps(model, data_iterable, nsamples, seqlen=4096):
    dev = model.device
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    layers = model.model.layers

    if isinstance(data_iterable, torch.Tensor):
        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, seqlen, nsamples)

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
    print(f'nsamples: {nsamples} seqlen: {seqlen} hidden_size: {model.config.hidden_size}')
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
        "position_ids",  # TODO: Remove
    ]

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
    print(f'config_type = {model.config.model_type}')
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def perplexity_eval(model, testenc, randomize, tokenizer):
    seqlen = testenc.shape[0]
    dev = model.device
    nsamples = 1

    # use_cache = True
    # model.config.use_cache = False

    # get_inps(model, testenc, args, dev="cpu", nsamples=nsamples)
    inps, forward_args = get_inps(
        model, testenc, 1, seqlen
    )

    if randomize:
        inps = torch.randn_like(inps)

    print(f'inps = {inps}')

    outs = torch.zeros_like(inps)
    for k, v in forward_args.items():
        forward_args[k] = v.to(dev) if isinstance(v, torch.Tensor) else v

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        layers[i] = layer
        inps, outs = outs, inps

    for i in range(nsamples):
        print(f'final layer = {inps[i].to(dev)}')
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        if lm_logits.isneginf().any() or lm_logits.isinf().any() or lm_logits.isnan().any():
            print('\n====================== HUGE ERROR =====================\n')
        print(f'logits={lm_logits.argmax(-1)}')
        print(tokenizer.batch_decode(lm_logits.argmax(-1), skip_special_tokens=True)[0])


if __name__ == "__main__":
    pretrained_model_path = sys.argv[1]
    uncompressed_model_path = sys.argv[2]
    compressed_model_path = sys.argv[3] # '/home/elvircrn/CLionProjects/spqr_kernel/data/output_float16' # sys.argv[2]
    calibration_dataset = sys.argv[4]
    dev = sys.argv[5]
    with torch.no_grad():
        p = Mode.CPU if dev == 'cpu' else Mode.CUDA
        model = LLama(pretrained_model_path, compressed_model_path, p)

        encoded_hello = model.tokenizer('Hello', return_tensors="pt").to(device=model.device).input_ids

        encoded_goodbye = model.tokenizer('Goodbye', return_tensors="pt").to(device=model.device).input_ids

        os.system('clear')
        text = input()
        model.generate(text)

        # perplexity_eval(model.model, encoded_hello, False, model.tokenizer)

        # perplexity_eval(model.model, encoded_goodbye, False)

        # perplexity_eval(model.model, encoded_hello, True)
        # perplexity_eval(model.model, encoded_goodbye, True)

# if __name__ == "__main__":
#     pretrained_model_path = sys.argv[1]
#     compressed_model_path = sys.argv[2]
#     calibration_dataset = sys.argv[3]
#
#     with torch.no_grad():
#         model = LLama(pretrained_model_path, compressed_model_path, Load.CUDA)
#         model.generate('Hello')
#
#     plt.show()
