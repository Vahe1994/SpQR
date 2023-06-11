import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama' and 'falcon' supported"


def get_model(model_path, dtype="auto"):
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,  # see https://stackoverflow.com/questions/76356591
    )
    model.seqlen = 2048
    return model


def move_head(model, device):
    if model.config.model_type == "llama":
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)
    elif model.config.model_type == "RefinedWebModel":
        if model.transformer.ln_f is not None:
            model.transformer.ln_f = model.transformer.ln_f.to(device)
        model.lm_head = model.lm_head.to(device)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def get_lm_logits(inps_, model):
    if model.config.model_type == "llama":
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "RefinedWebModel":
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
    elif model.config.model_type == "RefinedWebModel":
        return model.transformer.h
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_layers(module, layers=(nn.Conv2d, nn.Linear)):
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
    elif model.config.model_type == "RefinedWebModel":
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
