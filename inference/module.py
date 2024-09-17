import os
from enum import Enum

import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, DynamicCache, LlamaTokenizer, AutoConfig

import inference
from modelutils import suspend_nn_inits, get_layers, find_sublayers, \
    layer_weight_dequantization
from quant_groups import Quantizer, quantize

torch.set_printoptions(sci_mode=False)

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


class Mode(Enum):
    CPU_DEQUANTIZE = 1
    CUDA = 2
    CPU = 3
    QUANTIZE_NEAREST = 4
    CPU_DEQUANTIZE_ORIGINAL = 5


class LLama:
    def find_layers_to_quantize(self, mod, compressed_path, weights_to_quantize=None, layer_id=-1, parent_name=''):
        if weights_to_quantize is None:
            weights_to_quantize = []
        for name, m in mod.named_children():
            if name.isnumeric():
                layer_id = int(name)

            if isinstance(m, inference.SPQRModule):
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
                self.find_layers_to_quantize(m, compressed_path, weights_to_quantize, layer_id, name)

        return weights_to_quantize

    def change_tensor(self, w):
        mod, name, p = w

        if self.flag == Mode.CPU_DEQUANTIZE_ORIGINAL:
            model = torch.load(p)
            w = layer_weight_dequantization(model)
            m, n = w.shape[0], w.shape[1]
            ln = nn.Linear(in_features=n, out_features=m, dtype=torch.float32)
            ln.weight = torch.nn.Parameter(w, requires_grad=False)
            setattr(mod, name, ln)
        else:
            spqr_module: inference.SPQRModule = inference.load_compressed_tensor(p)
            if self.flag == Mode.CPU_DEQUANTIZE:
                ln = nn.Linear(in_features=spqr_module.n, out_features=spqr_module.m, dtype=torch.float32)
                ln.weight = torch.nn.Parameter(inference.spqr_dequantize_compressed(spqr_module).float(),
                                               requires_grad=False)
                setattr(mod, name, ln)
            elif self.flag == Mode.CUDA:
                spqr_module.name = p
                setattr(mod, name, spqr_module)

    def linear_to_spqr(self, model, quantized_model_path, device):
        weights_to_quantize = self.find_layers_to_quantize(model, quantized_model_path, layer_id=-1, parent_name='')

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
        if flag == Mode.CPU_DEQUANTIZE or flag == Mode.CPU or flag == Mode.QUANTIZE_NEAREST or flag == Mode.CPU_DEQUANTIZE_ORIGINAL:
            device = 'cpu'
        else:
            device = 'cuda'

        if flag == Mode.CPU_DEQUANTIZE or flag == Mode.CPU_DEQUANTIZE_ORIGINAL:
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

        if flag == Mode.QUANTIZE_NEAREST:
            self.quantize_nearest(self.model, self.device)
        elif flag != Mode.CPU:
            self.model = self.linear_to_spqr(self.model, quantized_model_path, self.device)

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_str, max_new_tokens):
        past_key_values = DynamicCache()
        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        generated_ids = inputs.input_ids

        cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=self.device)

        for _ in range(max_new_tokens):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cache_position=cache_position,
                                 past_key_values=past_key_values, use_cache=True)

            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

            cache_position = cache_position[-1:] + 1  # add one more position for the next token
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
