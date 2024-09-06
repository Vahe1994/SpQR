import os
import sys

import spqr


def sanity_check(compressed_path: str, uncompressed_path: str):
    """
    Check if the original matrix matches
    :param compressed_path:
    :param uncompressed_path:
    :return:
    """
    return True

if __name__ == '__main__':
    model_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(output_path, exist_ok=True)

    layers = os.listdir(model_path)
    layers.sort()

    model_args = spqr.ModelArgs(model_path)


    os.system(f'cp {os.path.join(model_path, 'args.pt')} {os.path.join(output_path, 'args.pt')}')
    os.system(f'cp {os.path.join(model_path, 'not_quantized_weights.pt')} {os.path.join(output_path, 'not_quantized_weights.pt')}')

    for layer_id in layers:
        folder = os.path.join(model_path, layer_id)
        output_folder = os.path.join(output_path, layer_id)

        if not os.path.isdir(folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        layers = os.listdir(folder)
        for layer in layers:
            # Load the original SPQR format.
            spqr_host = spqr.load_original_tensor(os.path.join(folder, layer), model_args)

            deq_w = spqr.spqr_dequantize(spqr_host)
            spqr_module = spqr.SPQRModule(spqr_host)
            deq_w_c = spqr.spqr_dequantize_compressed(spqr_module)


            tensor_path = f'{os.path.join(output_folder, layer)}.pth'

            # Dump the compressed version
            spqr.write_tensor(spqr_host, tensor_path)

