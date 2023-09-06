from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

from quant_groups import dequantize

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama' and 'falcon' supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")


@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def get_model(model_path, load_quantized=None, dtype="auto"):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # consider trust_remote_code=True
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype).eval()
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
        else:
            print("Loading pretrained model ...")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
    model.seqlen = 2048

    print("Model loaded sucessfully ...")

    return model


def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.config.model_type == "llama":
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    elif model.config.model_type.lower() in FALCON_TYPES:
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type == "llama":
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type == "llama":
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type == "llama":
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name)


def load_quantized_model(model, load_path):
    layers = get_layers(model)
    for i in trange(len(layers)):
        layer = layers[i]
        sub_layers = find_sublayers(layer)
        for name in sub_layers:
            quantized_params_dict = read_quant_weight_from_file(load_path, i, name)
            sub_layers[name].weight = nn.Parameter(
                layer_weight_dequantization(quantized_params_dict).to(sub_layers[name].weight.data.dtype)
            )
        layers[i] = layer
    model.load_state_dict(torch.load(load_path + "/not_quantized_weights.pt"), strict=False)
    return model


def layer_weight_dequantization(quantized_params_dict):
    out_dim, in_dim = quantized_params_dict["weight_shape"]
    blocksize = quantized_params_dict["blocksize"]
    keep_last_columns = quantized_params_dict["keep_last_columns"]
    reconstructed_weight = torch.zeros(quantized_params_dict["weight_shape"])
    block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
    block_start_iter = block_start_iter
    current_ind = 0

    for block_start in block_start_iter:
        block_end = min(block_start + blocksize, in_dim)
        for column_index in range(block_start, block_end):
            if column_index % quantized_params_dict["groupsize"] == 0:
                if quantized_params_dict["quant_layer_scale_qq_scale"]:
                    dequantize_zeros = dequantize(
                        quantized_params_dict["quant_layer_zeros"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_zero"][current_ind],
                    )
                    dequantize_scale = dequantize(
                        quantized_params_dict["quant_layer_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_zero"][current_ind],
                    )
                else:
                    dequantize_zeros = quantized_params_dict["quant_layer_zeros"][current_ind]
                    dequantize_scale = quantized_params_dict["quant_layer_scale"][current_ind]
                current_ind += 1

            reconstructed_weight[:, column_index] = dequantize(
                quantized_params_dict["quant_weights"][:, column_index].unsqueeze(1),
                dequantize_scale.reshape(-1, 1),
                dequantize_zeros.reshape(-1, 1),
            ).reshape_as(reconstructed_weight[:, column_index])
    reconstructed_weight = (
        reconstructed_weight * (quantized_params_dict["outliers_matrix"].to_dense().cpu() == 0)
        + quantized_params_dict["outliers_matrix"].to_dense().cpu()
    )
    invperm = torch.argsort(quantized_params_dict["perm"]).cpu()
    reconstructed_weight = reconstructed_weight[:, invperm]
    return reconstructed_weight
