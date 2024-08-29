import os
import sys

import spqr
import torch

import modelutils

if __name__ == '__main__':
    model_path = sys.argv[1]
    output_path = sys.argv[2]

    layers = os.listdir(model_path)
    layers.sort()

    model_args = spqr.ModelArgs(model_path)

    for layer_id in layers:
        folder = os.path.join(model_path, layer_id)
        output_folder = os.path.join(output_path, layer_id)

        if not os.path.isdir(folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        layers = os.listdir(folder)
        for layer in layers:
            p = os.path.join(folder, layer)
            m = torch.load(p, map_location='cpu')
            # Load the original SPQR format.
            spqr_host = spqr.load_original_tensor(p, model_args)

            deq_w = spqr.spqr_dequantize(spqr_host).float()
            spqr_module = spqr.SPQRModule(spqr_host)

            deq_w_c = spqr.spqr_dequantize_compressed(spqr_module).float()
            deq_w_o = modelutils.layer_weight_dequantization(m).float()


            print('deq_w_c')
            for i in range(32):
                for j in range(32):
                    print(f'{deq_w_c[ i][ j]:.2f} ', end='')
                print('')


            print('deq_w_o')
            for i in range(32):
                for j in range(32):
                    print(f'{deq_w_o[i][j]:.2f} ', end='')
                print('')

            assert(torch.allclose(deq_w_o, deq_w_c))
