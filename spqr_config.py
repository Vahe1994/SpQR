from dataclasses import dataclass

__all__ = ["QuantizationConfig"]


@dataclass
class QuantizationConfig:
    model_path: str = None
    dataset: str = "c4"
    custom_data_path: str = ""
    seed: int = 0
    nsamples: int = 128
    percdamp: float = 0.01
    nearest: bool = False
    wbits: int = 3
    groupsize: int = None
    permutation_order: str = "act_order"
    true_sequential: bool = False
    sym: bool = False
    perchannel: bool = False
    mse: bool = False
    qq_scale_bits: int = None
    round_zero: bool = False
    qq_zero_bits: int = None
    qq_zero_sym: bool = False
    qq_groupsize: int = 16
    qq_mse: bool = True
    outlier_threshold: float = float("inf")
    simplified_outliers: bool = False
    offload_activations: bool = False
    load: bool = False
    save: bool = False
    skip_out_loss: bool = False
    wandb: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
