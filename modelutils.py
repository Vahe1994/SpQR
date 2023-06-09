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
    if model.config.model_type == "RefinedWebModel":
        fix_rotary_embedding(model)
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


# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


class RotaryEmbedding(torch.nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype != emb.dtype:
                emb = emb.to(dtype)

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def fix_rotary_embedding(model):
    for module in model.modules():
        if hasattr(module, "maybe_rotary"):  # use for switch
            head_dim = module.head_dim
            module.maybe_rotary = RotaryEmbedding(head_dim)


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
