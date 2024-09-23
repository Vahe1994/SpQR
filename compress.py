import os
import sys

import torch

import inference

import modelutils



if __name__ == '__main__':
    uncompressed_model_path = sys.argv[1]
    compressed_model_path = sys.argv[2]

    report_errors = False

    os.makedirs(uncompressed_model_path, exist_ok=True)

    layers = os.listdir(uncompressed_model_path)

    model_args = inference.ModelArgs(uncompressed_model_path)

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
            spqr_host = inference.load_original_tensor(tensor_path, model_args)

            spqr_module = inference.SPQRModule(spqr_host)

            m = torch.load(tensor_path, map_location='cpu')

            if report_errors:
                deq_w_c = inference.spqr_dequantize_compressed(spqr_module).half()
                deq_w_o = modelutils.layer_weight_dequantization(m).half()
                max_abs_error = (deq_w_c - deq_w_o).abs()
                cnt = (max_abs_error != 0).sum()
                print(f'INFO: Maximum absolute conversion error: {max_abs_error} cnt {cnt} nnz {spqr_module.nnz}')
                d = (deq_w_c - deq_w_o).abs()
                import matplotlib.pyplot as plt;

                vis = (torch.nn.functional.normalize(d) * 255).unsqueeze(0).unsqueeze(0)
                vis = torch.nn.functional.max_pool2d(vis, kernel_size=4, stride=4)
                vis = vis.squeeze()
                plt.imshow(vis.cpu().numpy()); plt.axis('off'); plt.show()

            tensor_path = f'{os.path.join(output_folder, tensor_name)}.pth'

            # Dump the compressed version
            inference.write_tensor(spqr_module, tensor_path)
