import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import trange

from inference.module import LLama, Mode
from modelutils import get_lm_logits

torch.autograd.set_grad_enabled(False)

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
def perplexity_eval(model, testenc, randomize, tokenizer, dataset_name, nsamples):
    seqlen = 4096
    dev = model.device

    inps, forward_args = get_inps(
        model, testenc, nsamples, seqlen
    )

    if randomize:
        inps = torch.randn_like(inps)

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
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        print(tokenizer.batch_decode(lm_logits.argmax(-1), skip_special_tokens=True)[0])

    testenc = testenc.to(dev)

    nlls = []
    for i in trange(nsamples):
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"\n{dataset_name} perplexity = {ppl.item():.4f}\n")


def old():
    pretrained_model_path = sys.argv[1]
    # uncompressed_model_path = sys.argv[2]
    compressed_model_path = sys.argv[2]
    mode = sys.argv[3]
    # calibration_dataset = sys.argv[4]
    # dev = sys.argv[5]

    m = Mode(int(mode))

    with torch.no_grad():
        model = LLama(pretrained_model_path, compressed_model_path, m)
        text = 'The recipe for banana bread is '  # input()
        s = time.time();
        generated_text = model.generate(text, max_new_tokens=45);
        e = time.time();
        print(f'{generated_text}');
        print(f'Duration = {e - s}s')


if __name__ == "__main__":
    pretrained_model_path = sys.argv[1]
    # uncompressed_model_path = sys.argv[2]
    compressed_model_path = sys.argv[2]
    mode = sys.argv[3]
    # calibration_dataset = sys.argv[4]
    # dev = sys.argv[5]

    m = Mode(int(mode))

    with torch.no_grad():
        model = LLama(pretrained_model_path, compressed_model_path, m)
        text = 'The recipe for banana bread is '  # input()
        s = time.time();
        generated_text = model.generate(text, max_new_tokens=128);
        e = time.time();
        print(f'{generated_text}');
        print(f'Duration = {e - s}s')
