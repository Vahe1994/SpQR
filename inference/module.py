import os
import time
from enum import Enum, IntEnum

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, DynamicCache, LlamaTokenizer, AutoConfig, StaticCache, PretrainedConfig

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


class Mode(IntEnum):
    CPU_DEQUANTIZE = 1
    CUDA = 2
    CPU = 3
    QUANTIZE_NEAREST = 4
    CPU_DEQUANTIZE_ORIGINAL = 5
    CUDA_PT = 6
    CUDA_DENSE = 7


def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token


class LLama:
    def find_spqr_modules(self, mod, compressed_path, spqr_modules=None, layer_id=-1):
        if spqr_modules is None:
            spqr_modules = []

        if isinstance(mod, inference.SPQRModule):
            spqr_modules.append(mod)

        for name, m in mod.named_children():
            self.find_spqr_modules(m, compressed_path, spqr_modules, layer_id)

        return spqr_modules

    def find_layers_to_quantize(self, mod, compressed_path, parent_module=None, weights_to_quantize=None, layer_id=-1,
                                parent_name=''):
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
                    weights_to_quantize.append((parent_module, mod, name, p))
                elif os.path.exists(p_pt):
                    weights_to_quantize.append((parent_module, mod, name, p_pt))
                elif os.path.exists(p_pth):
                    weights_to_quantize.append((parent_module, mod, name, p_pth))
            elif m is not mod:
                self.find_layers_to_quantize(m, compressed_path, mod, weights_to_quantize, layer_id, name)

        return weights_to_quantize

    def change_tensor(self, w):
        parent_module, mod, name, p = w

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

        self.torchscript = False

        if flag == Mode.CUDA_PT:
            self.model = torch.load(quantized_model_path)
            spqr_modules = self.find_spqr_modules(self.model, quantized_model_path, layer_id=-1)
        else:
            with suspend_nn_inits():
                with torch.no_grad():
                    config = AutoConfig.from_pretrained(pretrained_model_path, torchscript=self.torchscript,
                                                        return_dict=True)
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
        elif flag != Mode.CPU and flag != Mode.CUDA_PT and flag != Mode.CUDA_DENSE:
            self.model = self.linear_to_spqr(self.model, quantized_model_path, self.device)

        # self.model =
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        # self.model = torch.compile(self.model.to(device=self.device, dtype=self.dtype), backend='cudagraphs')
        # self.model = torch.compile(self.model.to(device=self.device, dtype=self.dtype), backend='cudagraphs')
        # self.model = torch.compile(self.model.to(device=self.device, dtype=self.dtype))

        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_path, use_fast=False,
                                                        torchscript=self.torchscript)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(self, input_str, max_new_tokens):

        # self.model.generation_config.cache_implementation = "static"

        # self.model.forward = torch.compile(self.model.forward, backend="cudagraphs", fullgraph=False)

        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        input_ids = inputs.input_ids
        total_batch_duration = None
        seq_len = input_ids.shape[1]

        cache_position = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        generated_ids = torch.zeros(1, seq_len + max_new_tokens * 2, dtype=torch.int, device=self.device)
        generated_ids[:, cache_position] = input_ids.to("cuda").to(torch.int)

        past_key_values = StaticCache(self.model.config,
                                      1,
                                      seq_len + max_new_tokens * 2 + 1,
                                      device=self.device,
                                      dtype=torch.float16)
        # self.model = torch.jit.script(self.model)

        logits = self.model(
            input_ids, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_len]] = next_token

        torch._dynamo.config.capture_scalar_outputs = True

        with torch.no_grad():
            # Compile the CUDA graph
            decode_one_tokens_compiled = torch.compile(decode_one_tokens, mode='reduce-overhead', fullgraph=False)

            # Generate tokens one by one
            cache_position = torch.tensor([seq_len + 1], device="cuda")
            for _ in range(1, max_new_tokens):
                # with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    start_time = time.time()
                    next_token = decode_one_tokens_compiled(self.model, next_token.clone(), None, cache_position, past_key_values)
                    generated_ids[:, cache_position] = next_token.int()
                    end_time = time.time()
                    print(f'duration = {end_time - start_time}')

                cache_position += 1
                # print(self.tokenizer.decode(generated_ids[0]))

        return self.tokenizer.decode(generated_ids[0])
        #
        # for _ in range(max_new_tokens - 1):
        #     input_ids = inputs['input_ids']
        #     attention_mask = inputs['attention_mask']
        #
        #     dev = input_ids.device
        #
        #     start_time = time.time()
        #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, cache_position=cache_position,
        #                          past_key_values=past_key_values, use_cache=True, return_dict=False)
        #     # kwarg_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'cache_position': cache_position,
        #     #                 'past_key_values': past_key_values, 'use_cache': True, 'return_dict': True}
        #     # outputs = torch.jit.trace(self.model.forward, example_kwarg_inputs=kwarg_inputs)
        #     # outputs = self.model(input_ids=input_ids, example_kwarg_inputs=kwarg_inputs)
        #     torch.cuda.synchronize(dev)
        #     end_time = time.time()
        #     batch_duration = end_time - start_time
        #     print(f'duration = {batch_duration}')
        #
        #     if total_batch_duration is None:
        #         total_batch_duration = 0
        #     else:
        #         total_batch_duration += batch_duration
        #
        #     if self.torchscript:
        #         next_token_ids = outputs[0][:, -1:].argmax(-1)
        #     else:
        #         next_token_ids = outputs.logits[:, -1:].argmax(-1)
        #
        #     generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        #
        #     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        #     inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
        #
        #     cache_position = cache_position[-1:] + 1  # add one more position for the next token
        # print(f'Batches duration = {total_batch_duration}')
        # return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
