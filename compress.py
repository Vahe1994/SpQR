import os
import sys

import torch

import inference

import modelutils
from inference import SPQRUncompressed, list_flatten

sparse_compression_strategy = int(sys.argv[3])

class ModelArgs:
    bits: int
    beta1: int
    beta2: int

    def __init__(self, model_path: str):
        b = torch.load(os.path.join(model_path, 'args.pt'))
        self.bits = b['wbits']
        self.beta1 = b['qq_groupsize']
        self.beta2 = b['groupsize']


def load_uncompressed_spqr_tensor(p: str, model_args: ModelArgs) -> SPQRUncompressed:
    bits = model_args.bits
    beta1 = model_args.beta1
    beta2 = model_args.beta2

    t = torch.load(p, map_location='cpu')

    W = t['quant_weights']
    m = W.shape[0]
    n = W.shape[1]
    W = list_flatten(W)
    W_s = list_flatten(t['quant_layer_scale'])
    W_z = list_flatten(t['quant_layer_zeros'])

    perm = t['perm']

    outliers_matrix = t['outliers_matrix'].to_sparse_csr()

    col_ids = outliers_matrix.col_indices().short()
    values = outliers_matrix.values().half()

    return SPQRUncompressed(
        m=m,
        n=n,
        bits=bits,
        W=list_flatten(W),
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=list_flatten(t['quant_layer_scale_qq_scale']),
        W_s_z=list_flatten(t['quant_layer_scale_qq_zero']),
        W_z_s=list_flatten(t['quant_layer_zero_qq_scale']),
        W_z_z=list_flatten(t['quant_layer_zero_qq_zero']),
        row_offsets=outliers_matrix.crow_indices().int(),
        col_ids=col_ids,
        values=values,
        in_perm=perm.long(),
        sparse_compression_strategy=sparse_compression_strategy
    )


if __name__ == '__main__':
    uncompressed_model_path = sys.argv[1]
    compressed_model_path = sys.argv[2]

    report_errors = False

    os.makedirs(uncompressed_model_path, exist_ok=True)

    layers = os.listdir(uncompressed_model_path)

    model_args = ModelArgs(uncompressed_model_path)

    for f in ['args.pt', 'not_quantized_weights.pt']:
        os.system(f'cp {os.path.join(uncompressed_model_path, f)} {os.path.join(compressed_model_path, f)}')

    for layer_id in layers:
        folder = os.path.join(uncompressed_model_path, layer_id)
        output_folder = os.path.join(compressed_model_path, layer_id)

        if not os.path.isdir(folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        layers = os.listdir(folder)
        for tensor_name in layers:
            tensor_path = os.path.join(folder, tensor_name)
            print(f'INFO: Converting layer {layer_id}  tensor name = {tensor_name}')

            # Load the original SPQR format.
            spqr_host = load_uncompressed_spqr_tensor(tensor_path, model_args)

            spqr_module = inference.SPQRModule(spqr_host)

            m = torch.load(tensor_path, map_location='cpu')

            if report_errors:
                deq_w_c = inference.spqr_dequantize_compressed(spqr_module).half()
                deq_w_o = modelutils.layer_weight_dequantization(m).half()
                max_abs_error = (deq_w_c - deq_w_o).abs()
                cnt = (max_abs_error != 0).sum()
                print(f'INFO: Maximum absolute conversion error: {max_abs_error} cnt {cnt} nnz {spqr_module.nnz}')
                d = (deq_w_c - deq_w_o).abs()
                import matplotlib.pyplot as plt

                vis = (torch.nn.functional.normalize(d) * 255).unsqueeze(0).unsqueeze(0)
                vis = torch.nn.functional.max_pool2d(vis, kernel_size=4, stride=4)
                vis = vis.squeeze()
                plt.imshow(vis.cpu().numpy())
                plt.axis('off')
                plt.show()

            tensor_path = f'{os.path.join(output_folder, tensor_name)}.pth'

            # Dump the compressed version
            inference.write_tensor(spqr_module, tensor_path)
