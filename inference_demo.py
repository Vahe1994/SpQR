import time
from enum import IntEnum
from typing import Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, StaticCache

from modelutils import suspend_nn_inits

torch.autograd.set_grad_enabled(False)

torch.set_printoptions(sci_mode=False)

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


class Mode(IntEnum):
    DENSE = 0
    QUANTIZED = 1
    TORCH_PT = 2


def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True,
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token


class InferenceDemo:
    def __init__(
        self, pretrained_model_path: str, quantized_model_path, flag, device="cuda", torchscript=False, backend=None
    ):
        self.flag = flag
        self.device = device
        self.dtype = torch.float16
        self.torchscript = torchscript
        self.backend = backend

        if flag == Mode.TORCH_PT:
            self.config = AutoConfig.from_pretrained(pretrained_model_path, torchscript=self.torchscript)
            self.model = torch.load(quantized_model_path)
        elif flag == Mode.QUANTIZED:
            with suspend_nn_inits():
                with torch.no_grad():
                    self.config = AutoConfig.from_pretrained(
                        quantized_model_path, torchscript=self.torchscript, return_dict=True, from_tf=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=quantized_model_path,
                        trust_remote_code=True,
                        config=self.config,
                        from_tf=False,
                        weights_only=False,
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                    )
                    print("Finished loading")
        else:
            with suspend_nn_inits():
                with torch.no_grad():
                    self.config = AutoConfig.from_pretrained(
                        pretrained_model_path, torchscript=self.torchscript, return_dict=True
                    )

                    self.model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.half,
                        config=self.config,
                    )

        if self.torchscript:
            self.model = torch.jit.script(self.model)

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, torchscript=self.torchscript
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(self, input_str, max_new_tokens) -> Tuple:
        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        cache_position = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        generated_ids = torch.zeros(1, seq_len + max_new_tokens * 2, dtype=torch.int, device=self.device)
        generated_ids[:, cache_position] = input_ids.to(self.device).to(torch.int)

        past_key_values = StaticCache(
            self.model.config, 1, seq_len + max_new_tokens * 2 + 1, device=self.device, dtype=torch.float16
        )

        logits = self.model(
            input_ids, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_len]] = next_token

        torch._dynamo.config.capture_scalar_outputs = True

        forward_time_s = []
        with torch.no_grad():
            # Compile the CUDA graph
            if self.backend is None:
                decode_one_tokens_compiled = decode_one_tokens
            else:
                decode_one_tokens_compiled = torch.compile(decode_one_tokens, mode="default", fullgraph=True)

            # Generate tokens one by one
            cache_position = torch.tensor([seq_len + 1], device="cuda")
            for _ in range(1, max_new_tokens):
                start_time = time.time()
                next_token = decode_one_tokens_compiled(
                    self.model, next_token.clone(), None, cache_position, past_key_values
                )
                generated_ids[:, cache_position] = next_token.int()
                end_time = time.time()
                print(f"duration = {end_time - start_time}")
                forward_time_s.append(end_time - start_time)

                cache_position += 1

        return self.tokenizer.decode(generated_ids[0]), forward_time_s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to the model to the pretrained model",
    )
    parser.add_argument(
        "--compressed_model_path",
        type=str,
        help="Path to the compressed .pt model",
    )
    parser.add_argument(
        "--execution_mode",
        choices=[0, 1, 2],
        required=True,
        type=int,
        help="If set to 0, will evaluate the dense pretrained model. "
        "If set to 1, will evaluate the spqr-quantized model using HF"
        "If set to 2, will evaluate the spqr-quantized model using torch .pt",
    )

    args = parser.parse_args()

    m = Mode(args.execution_mode)

    with torch.no_grad():
        model = InferenceDemo(args.pretrained_model_path, args.compressed_model_path, m)
        text = "The recipe for banana bread is "  # input()
        s = time.time()
        generated_text, timings_s = model.generate(text, max_new_tokens=128)
        e = time.time()
        print(f"{generated_text}")

        print(f"Total duration = {e - s}s")

        durations = np.array(timings_s[16:])

        print(f'Mean duration after caching initial input = {durations.mean()}')
        print(f'Median duration after caching initial input = {np.median(durations)}')
        print(f'Best duration after caching initial input = {np.min(durations)}')
