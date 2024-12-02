import json
import os
import re
import shutil

import torch
from safetensors.torch import save_model
from spqr_quant import QuantizedLinear
from transformers import AutoConfig, AutoTokenizer


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def pack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    data[data >= 2 ** (nbits - 1)] -= 2**nbits
    return data.to(get_int_dtype(nbits))


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")


def get_layers_prefix(config) -> str:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")


def get_converted_state_dict(config, nbits: int, in_path: os.PathLike) -> [dict, list[str]]:
    state_dict = {}
    modules_to_not_convert = []

    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)

    for i in range(num_layers):
        layer = torch.load(os.path.join(in_path, f"{i}.pth"))
        for name, p in layer.named_parameters():
            if torch.is_floating_point(p.data):
                p.data = p.data.half()
            else:
                p.data = pack_int_data(p.data, nbits)
            if "quantized_weight." not in name:
                modules_to_not_convert.append(f"{layers_prefix}.{i}.{name}")
            else:
                name = re.sub("quantized_weight.", "", name)
            state_dict[f"{layers_prefix}.{i}.{name}"] = p.data

    for key, value in torch.load(os.path.join(in_path, "not_quantized_weights.pt")).items():
        state_dict[key] = value.half()
        modules_to_not_convert.append(key)

    if "lm_head.weight" not in modules_to_not_convert:
        modules_to_not_convert.append("lm_head.weight")

    return state_dict, modules_to_not_convert


def get_metadata(args_path: str) -> dict:
    quant_args = torch.load(args_path)
    return {"bits": quant_args["wbits"], "beta1": quant_args["qq_groupsize"], "beta2": quant_args["groupsize"]}


def update_config(config_dict: dict, spqr_metadata: dict[str, int], modules_to_not_convert: list[str]):
    config_dict["quantization_config"] = {
        "quant_method": "spqr",
        "beta1": spqr_metadata["beta1"],
        "beta2": spqr_metadata["beta2"],
        "bits": spqr_metadata["bits"],
    }
    config_dict["torch_dtype"] = None
    config_dict["_attn_implementation_autoset"] = False
    config_dict["architectures"] = []
    return config_dict


def add_inference_code(model_type: str, save_path: os.PathLike):
    if os.path.isdir(f"./transformers/{model_type}"):
        shutil.copytree(f"./transformers/{model_type}", save_path, dirs_exist_ok=True)
    else:
        print(f"No predefined PreTrainedModel exists for {model_type}. You'll have to copy-paste some code yourself.")


def replace_with_spqr_linear(
    model,
    quantization_config_shapes=None,
    modules_to_not_convert=None,
    current_key_name=None,
):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, QuantizedLinear):
            # Check if the current key is not in the `modules_to_not_convert`
            if ".".join(current_key_name) + ".weight" not in modules_to_not_convert:
                tensor_name = ".".join(current_key_name)
                quantization_config_shapes[f"{tensor_name}.dense_weights.shape"] = module.dense_weights.shape[0]
                quantization_config_shapes[f"{tensor_name}.row_offsets.shape"] = module.row_offsets.shape[0]
                quantization_config_shapes[f"{tensor_name}.col_vals.shape"] = module.col_vals.shape[0]
                quantization_config_shapes[f"{tensor_name}.in_perm.shape"] = module.in_perm.shape[0]
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_spqr_linear(
                module,
                quantization_config_shapes=quantization_config_shapes,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model to base config on, as in AutoConfig.from_pretrained()",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the model to base config on, as in AutoConfig.from_pretrained()",
    )
    parser.add_argument(
        "--in_path_pt",
        type=str,
        help="Path of the checkpoint to convert",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to save HF compatible checkpoint to",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Whether to save in safetensors format",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Whether to load model",
    )
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        help="Whether to save tokenizer",
    )
    args = parser.parse_args()

    old_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    metadata = get_metadata(os.path.join(args.config_path, "args.pt"))
    modules_to_not_convert = torch.load(os.path.join(args.config_path, "not_quantized_weights.pt")).keys()

    model = torch.load(args.in_path_pt)

    # convert to safetensors
    if args.save_safetensors:
        # load dummy model
        # torch.save(model, os.path.join(args.out_path, "pytorch_model.bin"))
        save_model(model, os.path.join(args.out_path, "model.safetensors"), metadata={"format": "pt"})

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.out_path)

    new_config_dict = update_config(old_config.to_diff_dict(), metadata, list(modules_to_not_convert))
    new_config_dict["quantization_config"]["modules_to_not_convert"] = list(modules_to_not_convert)

    new_config_dict["quantization_config"]["shapes"] = {}
    replace_with_spqr_linear(model, new_config_dict["quantization_config"]["shapes"], set(modules_to_not_convert))
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(new_config_dict, config_file, indent=4)
