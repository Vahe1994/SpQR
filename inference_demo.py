import os
import subprocess
import sys
from enum import Enum

import spqr
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, DynamicCache, LlamaTokenizer, AutoConfig

from datautils import get_loaders
from modelutils import suspend_nn_inits, get_layers, get_model_head, get_lm_logits, find_sublayers, \
    layer_weight_dequantization
from quant_groups import Quantizer, quantize

torch.set_printoptions(sci_mode=False)

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


class Load(Enum):
    CPU_DEQUANTIZE = 1
    CUDA = 2
    CPU = 3
    QUANTIZE_NEAREST = 4
    CPU_DEQUANTIZE_ORIGINAL = 5


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



class LLama:
    def _linear_to_spqr(self, mod, compressed_path, weights_to_quantize, layer_id=-1, parent_name=''):
        for name, m in mod.named_children():
            if name.isnumeric():
                layer_id = int(name)

            if isinstance(m, spqr.SPQRModule):
                continue

            if isinstance(m, torch.nn.Linear):
                assert (m.bias is None)
                p = os.path.join(compressed_path, f'{layer_id}', f'{parent_name}.{name}')
                p_pt = os.path.join(compressed_path, f'{layer_id}', f'{parent_name}.{name}.pt')
                p_pth = os.path.join(compressed_path, f'{layer_id}', f'{parent_name}.{name}.pth')
                if os.path.exists(p):
                    weights_to_quantize.append((mod, name, p))
                elif os.path.exists(p_pt):
                    weights_to_quantize.append((mod, name, p_pt))
                elif os.path.exists(p_pth):
                    weights_to_quantize.append((mod, name, p_pth))
            elif m is not mod:
                self._linear_to_spqr(m, compressed_path, weights_to_quantize, layer_id, name)

        return mod

    def change_tensor(self, w):
        flag = self.flag
        mod, name, p = w

        if flag == Load.CPU_DEQUANTIZE_ORIGINAL:
            model = torch.load(p)
            w = layer_weight_dequantization(model)
            m, n = w.shape[0], w.shape[1]
            ln = nn.Linear(in_features=n, out_features=m, dtype=torch.float32)
            ln.weight = torch.nn.Parameter(w, requires_grad=False)
            setattr(mod, name, ln)
        else:
            spqr_module: spqr.SPQRModule = spqr.load_compressed_tensor(p)
            if flag == Load.CPU_DEQUANTIZE:
                # FLOAT32
                ln = nn.Linear(in_features=spqr_module.n, out_features=spqr_module.m, dtype=torch.float32)
                ln.weight = torch.nn.Parameter(spqr.spqr_dequantize_compressed(spqr_module).float(), requires_grad=False)
                setattr(mod, name, ln)
            elif flag == Load.CUDA:
                spqr_module.name = p
                setattr(mod, name, spqr_module)

    def linear_to_spqr(self, model, quantized_model_path, device):
        weights_to_quantize = []
        model = self._linear_to_spqr(model, quantized_model_path, weights_to_quantize, layer_id=-1, parent_name='')

        for w in tqdm(weights_to_quantize, "Loading quantized model"):
            self.change_tensor(w)

        model.load_state_dict(torch.load(quantized_model_path + "/not_quantized_weights.pt"), strict=False)
        final_model = model.to(device)
        return final_model
    @torch.no_grad()
    def quantize_nearest(self, model, dev):
        wbits = 3
        """Round-to-nearest quantization"""
        layers = get_layers(model)
        for i in trange(len(layers), desc="quantizing layers to nearest"):
            layer_dev = next(layers[i].parameters()).device
            layer = layers[i].to(dev)
            subset = find_sublayers(layer)
            for name in subset:
                quantizer = Quantizer()

                # m, n = subset[name].weight
                # tile_m =

                # quantizer.configure(wbits, perchannel=True, sym=False,qq_scale_bits=)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                    next(iter(layer.parameters())).dtype
                )
            layers[i] = layer.to(layer_dev)
            del layer
            torch.cuda.empty_cache()
        return None, wbits

    def __init__(self, pretrained_model_path: str, quantized_model_path, flag):
        self.flag = flag
        if flag == Load.CPU_DEQUANTIZE or flag == Load.CPU or flag == Load.QUANTIZE_NEAREST or flag == Load.CPU_DEQUANTIZE_ORIGINAL:
            device = 'cpu'
        else:
            device = 'cuda'

        if flag == Load.CPU_DEQUANTIZE or flag == Load.CPU_DEQUANTIZE_ORIGINAL:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self.device = device
        with suspend_nn_inits():
            config = AutoConfig.from_pretrained(pretrained_model_path)
            config.max_position_embeddings = 4096

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_path,
                trust_remote_code=True,
                torch_dtype=torch.half,
                config=config
            )

        self.device = device

        if flag == Load.QUANTIZE_NEAREST:
            self.quantize_nearest(self.model, self.device)
        elif flag != Load.CPU:
            self.model = self.linear_to_spqr(self.model, quantized_model_path, self.device)

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_str):
        past_key_values = DynamicCache()
        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        # generated_ids = inputs.input_ids
        generated_ids = inputs.input_ids
        # generated_ids = torch.flip(generated_ids, [0, 1])

        cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=self.device)
        max_new_tokens = 100

        for _ in range(max_new_tokens):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cache_position=cache_position,
                                 past_key_values=past_key_values, use_cache=True)

            # Greedily sample one next token
            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            # generated_ids = torch.flip(generated_ids, [0, 1])

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

            cache_position = cache_position[-1:] + 1  # add one more position for the next token
            os.system('clear')
            print(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    pretrained_model_path = sys.argv[1]
    uncompressed_model_path = sys.argv[2]
    compressed_model_path = sys.argv[3] # '/home/elvircrn/CLionProjects/spqr_kernel/data/output_float16' # sys.argv[2]
    calibration_dataset = sys.argv[4]
    dev = sys.argv[5]
    with torch.no_grad():
        p = Load.CPU if dev == 'cpu' else Load.CUDA
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
