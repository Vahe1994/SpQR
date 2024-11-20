import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from inference_lib.spqr_quant.inference import ModelArgs, SPQRLegacy, QuantizedLinear


def load_legacy_tensor(p: str, model_args: ModelArgs) -> SPQRLegacy:
    """
    Load legacy tensor given tensor path @p and model args @model_args.
    Background:
    spqr_engine.py produces tensors whose 3-bit weights are stored as int8.
    We refer to this storage scheme as legacy, since the 3-bit inference kernel
    only accepts the compressed storage format.
    @param p: Legacy tensor path.
    @param model_args: Model arguments - we obtain the beta1, beta2, bits and the sparse compression format from here.
    @return: QuantizedLinear object, storing the compressed matrix format ready to be used by the efficient inference
    kernel.
    """

    def flatten_tensor(W):
        """
        @return: Utility function: flattens the input tensor.
        """
        if torch.is_tensor(W):
            return W.flatten()
        else:
            return torch.cat(W).flatten()

    bits = model_args.bits
    beta1 = model_args.beta1
    beta2 = model_args.beta2

    legacy_tensor = torch.load(p, map_location="cpu")

    W = legacy_tensor["quant_weights"]
    m = W.shape[0]
    n = W.shape[1]
    W = flatten_tensor(W)
    W_s = flatten_tensor(legacy_tensor["quant_layer_scale"])
    W_z = flatten_tensor(legacy_tensor["quant_layer_zeros"])

    perm = legacy_tensor["perm"]

    outliers_matrix = legacy_tensor["outliers_matrix"].to_sparse_csr()

    col_ids = outliers_matrix.col_indices().short()
    values = outliers_matrix.values().half()

    return SPQRLegacy(
        m=m,
        n=n,
        bits=bits,
        W=flatten_tensor(W),
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=flatten_tensor(legacy_tensor["quant_layer_scale_qq_scale"]),
        W_s_z=flatten_tensor(legacy_tensor["quant_layer_scale_qq_zero"]),
        W_z_s=flatten_tensor(legacy_tensor["quant_layer_zero_qq_scale"]),
        W_z_z=flatten_tensor(legacy_tensor["quant_layer_zero_qq_zero"]),
        row_offsets=outliers_matrix.crow_indices().int(),
        col_ids=col_ids,
        values=values,
        in_perm=perm.long(),
    )


def replace_and_save_quantized_layers(
    model_args: ModelArgs,
    model_to_be_quantized,
    legacy_model_path,
    current_model=None,
    layer_id: int = -1,
    parent_tensor_name="",
    output_per_layer_path=None,
):
    """
    This function goes through the @model_to_be_quantized recursively and
    replaces all the dense layers with their quantized counterpart where
    applicable. The legacy quantized layers are stored in @legacy_model_path.

    As we go through the model, we construct the tensor name using layer_id and parent tensor name.
    We then use these values to check if the current dense tensor is a valid candidate for substitution
    with its quantized counterpart.

    @param model_args: Global model args.
    @param model_to_be_quantized: Model to be quantized.
    @param legacy_model_path: Location of the quantized tnesors stored in the legacy format as output by SpQR.
    @param output_per_layer_path: Optionally, one may wish to store the compressed SpQR layers separately in a folder
    specified by this parameter (for example, this may or may not be useful during benchmarking or data analysis).
    @param layer_id: Internal used to keep track of the current layer as we descend the model.
    @param parent_tensor_name: Name of the previous layer in the recursion chain.
    """
    if current_model == None:
        current_model = model_to_be_quantized
    for tensor_name, m in current_model.named_children():
        if tensor_name.isnumeric():
            layer_id = int(tensor_name)
            if output_per_layer_path is not None:
                os.makedirs(os.path.join(output_per_layer_path, str(layer_id)), exist_ok=True)

        if isinstance(m, torch.nn.Linear):
            assert m.bias is None
            legacy_tensor_path = os.path.join(legacy_model_path, f"{layer_id}", f"{parent_tensor_name}.{tensor_name}")
            if os.path.exists(legacy_tensor_path):
                spqr_uncompressed = load_legacy_tensor(legacy_tensor_path, model_args)
                spqr_module = QuantizedLinear.from_legacy(spqr_uncompressed, model_args, "cpu")
                if output_per_layer_path is not None:
                    per_layer_tensor_path = os.path.join(
                        output_per_layer_path, f"{layer_id}", f"{parent_tensor_name}.{tensor_name}"
                    )
                    torch.save(spqr_module, per_layer_tensor_path)
                setattr(current_model, tensor_name, spqr_module)
        else:
            replace_and_save_quantized_layers(
                model_args, model_to_be_quantized, legacy_model_path, m, layer_id, tensor_name, output_per_layer_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the unquantized model",
    )
    parser.add_argument(
        "--legacy_model_path",
        type=str,
        required=True,
        help="path to legacy model",
    )
    parser.add_argument(
        "--sparse_strategy",
        type=str,
        default="csr",
        choices=["csr", "ptcsr", "optimize_latency"],
        help="Sparse strategy storage. Options: csr, ptcsr, auto.\nCSR - Compressed Sparse Rows\nPTCSR - Alternative storage format\noptimize_latency - Use the current GPU to determine the optimal storage format to reduce kernel latency",
    )
    parser.add_argument("--save_pt", type=str, required=False, help="Save the converted quantized .pt model here")
    parser.add_argument(
        "--save_per_layer",
        type=str,
        required=False,
        help="Save the converted quantized model per layer here - useful for benchmarking individual layers",
    )

    args, leftovers = parser.parse_known_args()

    if args.save_per_layer is not None:
        os.makedirs(args.save_per_layer, exist_ok=True)

    layers = os.listdir(args.legacy_model_path)

    args_path = os.path.join(args.legacy_model_path, "args.pt")
    model_args = ModelArgs.from_file(args.legacy_model_path, args.sparse_strategy)

    config = AutoConfig.from_pretrained(args.base_model, return_dict=True)

    config.max_position_embeddings = 4096

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model, trust_remote_code=True, torch_dtype=torch.half, config=config
    )

    if args.save_per_layer is not None:
        not_quantized_weights_path = os.path.join(args.legacy_model_path, "not_quantized_weights.pt")
        not_quantized_weights = torch.load(not_quantized_weights_path)
        for w in not_quantized_weights.values():
            w.requires_grad = False
        model.load_state_dict(not_quantized_weights, strict=False)
        for f in ["args.pt", "not_quantized_weights.pt"]:
            os.system(f"cp {os.path.join(args.legacy_model_path, f)} {os.path.join(args.save_per_layer, f)}")

    replace_and_save_quantized_layers(
        model_args, model, args.legacy_model_path, output_per_layer_path=args.save_per_layer
    )

    if args.save_pt is not None:
        torch.save(model, args.save_pt)
