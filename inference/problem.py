import os
import time

import numpy as np
import spqr_cuda
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix

from inference import SPQRHost, updiv, compress, list_flatten, random_like


def max_pool2d(input_tensor, kernel_size):
    """
    Perform max pooling on a 2D tensor using PyTorch's built-in function.

    Args:
    - input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    - kernel_size (int): Size of the kernel for max pooling.

    Returns:
    - output_tensor (torch.Tensor): Output tensor after max pooling.
    """

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # Apply max pooling using torch.nn.functional.max_pool2d
    output_tensor = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=kernel_size)

    return output_tensor.squeeze()



def generate_sparse_matrix(total_rows, total_cols, sub_tile_size, sparsity):
    # Calculate the number of sub-tiles in each dimension
    num_sub_tiles_rows = total_rows // sub_tile_size
    num_sub_tiles_cols = total_cols // sub_tile_size

    # Initialize an empty matrix
    sparse_matrix = np.zeros((total_rows, total_cols))

    # Generate each sub-tile
    for i in range(num_sub_tiles_rows):
        for j in range(num_sub_tiles_cols):
            # Create a sparse 16x16 sub-tile
            sub_tile = np.random.choice([0, 1], size=(sub_tile_size, sub_tile_size), p=[sparsity, 1 - sparsity])
            # Place the sub-tile into the appropriate location in the full matrix
            sparse_matrix[i * sub_tile_size:(i + 1) * sub_tile_size,
            j * sub_tile_size:(j + 1) * sub_tile_size] = sub_tile

    return sparse_matrix


class Problem:
    def __init__(self, p, device: torch.device, randomize_sparse: bool = False,
                 sort_nnz: bool = False, generate_tiles: bool = False):
        t = torch.load(p, map_location='cpu')

        self.W = t['quant_weights']
        self.m = self.W.shape[0]
        self.n = self.W.shape[1]
        self.W = list_flatten(self.W)
        self.W_s = list_flatten(t['quant_layer_scale'])
        self.W_z = list_flatten(t['quant_layer_zeros'])
        self.W_s_s = list_flatten(t['quant_layer_scale_qq_scale'])
        self.W_s_z = list_flatten(t['quant_layer_scale_qq_zero'])
        self.W_z_s = list_flatten(t['quant_layer_zero_qq_scale'])
        self.W_z_z = list_flatten(t['quant_layer_zero_qq_zero'])

        total_blocks = updiv(self.m, self.args.beta1) * updiv(self.n, self.args.beta2)

        outliers_matrix = t['outliers_matrix']
        outliers_matrix = outliers_matrix.to_sparse_csr()

        self.row_offsets = outliers_matrix.crow_indices().cuda(device=device).int()
        self.col_ptr = outliers_matrix.col_indices().cuda(device=device).short()
        self.values = outliers_matrix.values().cuda(device=device).half()

        row_nnzs = np.diff(outliers_matrix.crow_indices())
        sorted_row_indices = np.argsort(-row_nnzs)
        self.values = torch.ones_like(self.values)

        self.h_row_offsets = self.row_offsets.int().cpu()
        self.h_col_ptr = self.col_ptr.short().cpu()
        self.h_values = self.values.float().cpu()

        csrmat = csr_matrix((self.h_values, self.h_col_ptr, self.h_row_offsets), shape=(self.m, self.n))

        densemat = csrmat.todense()
        if randomize_sparse:
            densemat = generate_sparse_matrix(densemat.shape[0], densemat.shape[1], 16, 0.5)
        else:
            if sort_nnz:
                densemat = densemat[sorted_row_indices]
            self.nnz = outliers_matrix._nnz()

        bsr = torch.tensor(densemat, dtype=torch.float32).to_sparse_bsr(blocksize=(self.args.beta1, self.args.beta2))
        self.nnz = 0
        self.first_order_data = torch.zeros(32 * total_blocks, dtype=torch.int32, device='cpu')
        self.second_order_data = torch.zeros(4 * total_blocks, dtype=torch.short, device='cpu')

        if generate_tiles:
            coos = [coo_matrix(b) for b in bsr.values()]
            self.tile_row_offsets = bsr.crow_indices().int().cuda(device=device)
            self.tile_col_ptr = bsr.col_indices().short().cuda(device=device)
            self.tile_count = bsr.values().shape[0]
            self.nnz = 0
            for coo in coos:
                self.nnz += coo.data.shape[0]
            self.tile_nnzs = torch.zeros(self.tile_row_offsets[-1] + 1).int()
            tiles_data = np.zeros(self.nnz, dtype=np.int32)
            nnz_id = 0
            for i, m in enumerate(bsr.values()):
                tile_col = bsr.col_indices()[i].item() * self.args.beta2

                coo = coo_matrix(m)

                nnz = coo.data.shape[0]
                assert (nnz != 0)
                assert (coo.data.shape == (coo.data != 0).shape)
                self.tile_nnzs[i + 1] = nnz_id + nnz
                sparsity_half_vals = torch.tensor(coo.data).half()
                sparsity_half_vals = torch.ones_like(sparsity_half_vals)

                coo_cols = (tile_col + coo.col) & ((1 << 12) - 1)

                tiles_data[nnz_id:(nnz_id + nnz)] = \
                    np.bitwise_or(sparsity_half_vals.view(torch.int16).int().numpy().astype(np.uint32) << 16,
                                  np.bitwise_or(coo_cols << 4, (coo.row & ((1 << 4) - 1)))
                                  .astype(np.uint32))
                nnz_id += nnz
                self.tile_data = torch.tensor(tiles_data, dtype=torch.int32).cuda(device=device)
            self.tile_row_count = self.tile_row_offsets.shape[0] - 1
            self.tile_nnzs = self.tile_nnzs.cuda(device=device)

            plot_tiles = False
            if plot_tiles:
                nnzs = np.diff(self.tile_nnzs)
                plt.hist(nnzs, bins=np.arange(nnzs.min(), nnzs.max() + 1))
                plt.show()

        self.tensor_compress_interleaved()
        self.second_order_data = self.second_order_data.cuda(device=device)
        self.first_order_data = self.first_order_data.cuda(device=device)
        self.x = torch.ones(self.n).cuda(device=device).half()
        self.y = torch.zeros(self.m).cuda(device=device).half()

    def analyse_sparsity(self, dense, name, base_path):
        plt.axis('off')
        d = max_pool2d(dense, 16)
        plt.imshow(d)
        # plt.tight_layout()
        d = max_pool2d(dense, 16)
        perc_tiles_filled = ((d.flatten() > 0).sum() / d.flatten().shape[0])
        density = ((dense.flatten() > 0).sum() / dense.flatten().shape[0])
        Round = lambda x, n: eval(
            '"%.' + str(int(n)) + 'f" % ' + repr(int(x) + round(float('.' + str(float(x)).split('.')[1]), n)))
        plt.savefig(
            os.path.join(base_path,
                         f'visualization/{name}_ratiofilled={Round(perc_tiles_filled, 2)}_density={Round(density, 2)}.png'),
            bbox_inches='tight', transparent=True, pad_inches=0, dpi=400)

    def run(self, num_runs, feature_flag):
        runs = torch.empty(num_runs).cpu().float()

        for i in range(num_runs):
            self.y.zero_()
            spqr_cuda.spqr_mul_timer(
                self.m,
                self.n,
                self.args.bits,
                self.args.beta1,
                self.args.beta2,
                self.first_order_data,
                self.second_order_data,
                self.values,
                self.row_offsets,
                self.col_ptr,
                self.tile_row_count,
                self.tile_count,
                self.tile_row_offsets,
                self.tile_col_ptr,
                self.tile_nnzs,
                self.tile_data,
                self.x,
                self.y,
                self.nnz,
                runs[i],
                feature_flag)

        time.sleep(1)
        return runs, self.y.clone()

    def compress_row(self, w, ws, wz):
        pass

    def tensor_compress_interleaved(self):
        spqr_cuda.tensor_compress_interleaved(
            self.m,
            self.n,
            self.args.bits,
            self.W,
            self.args.beta1,
            self.args.beta2,
            self.W_s,
            self.W_z,
            self.W_s_s,
            self.W_s_z,
            self.W_z_s,
            self.W_z_z,
            self.first_order_data,
            self.second_order_data
        )


def ones(m, n, beta1=16, beta2=16, bits=3) -> SPQRHost:
    m = m
    n = n
    bits = bits
    beta1 = beta1
    beta2 = beta2

    c = lambda E: compress(E, bits)

    W_uncompressed = torch.ones(m, n).int()
    W = c(W_uncompressed).int()

    num_first_order_groups = updiv(m, beta1) * n
    num_second_order_groups = updiv(m, beta1) * updiv(n, beta2)

    W_s_raw = torch.ones(num_first_order_groups).char().cpu()
    W_z_raw = torch.zeros(num_first_order_groups).char().cpu()
    W_s = c(W_s_raw.int())
    W_z = c(W_z_raw.int())

    W_s_s = torch.ones(num_second_order_groups).float()
    W_s_z = torch.zeros(num_second_order_groups).float()
    W_z_s = torch.zeros(num_second_order_groups).float()
    W_z_z = torch.zeros(num_second_order_groups).float()

    values = torch.zeros(1).float()
    row_offsets = torch.zeros(m + 1).int()
    col_ids = torch.zeros(1)
    nnz = 0

    x = torch.ones(n).float()
    y_gt = torch.zeros(m).float()
    y = torch.zeros(m).float()

    return SPQRHost(
        m=m,
        n=n,
        bits=bits,
        W=W,
        beta1=beta1,
        beta2=beta2,
        W_s=W_s,
        W_z=W_z,
        W_s_s=W_s_s,
        W_s_z=W_s_z,
        W_z_s=W_z_s,
        W_z_z=W_z_z,
        row_offsets=row_offsets,
        col_ids=col_ids,
        values=values,
        nnz=nnz,
        W_dequantized=W_uncompressed.char(),
        W_s_raw=W_s_raw,
        W_z_raw=W_z_raw)
