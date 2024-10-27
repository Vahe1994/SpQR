import sys

import inference

import plotly.graph_objects as go
import plotly.express as px

colors = px.colors.sequential.Plasma
num_colors = len(colors)
colors = [colors[0], colors[num_colors // 2 - 1], colors[-4]]


import os
import time

import numpy as np
import spqr_cuda
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix

from inference import updiv, list_flatten


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


class SparsityAnalysis:
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





def updiv(x, y): return (x + y - 1) // y


def densities():
    fig = go.Figure()
    device = torch.device('cuda:0')

    base_path = sys.argv[1]
    folders = os.listdir(base_path)
    args_path = os.path.join(base_path, 'args.pt')

    fig.update_yaxes(type='log')
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_layout(title='Densities', title_x=0.5, xaxis_title='Row (normalized)',
                      yaxis_title='Density ratio (log)')
    fig.show()

    for layer_id in folders:
        folder = os.path.join(base_path, layer_id)
        if not os.path.isdir(folder):
            continue
        for p in os.listdir(folder):
            tensor_name = f'L{layer_id}_{p}'
            prob = SparsityAnalysis(os.path.join(folder, p), device=device, randomize_sparse=False, sort_nnz=True, generate_tiles=False)

            r = 1
            spqr_host, spqr_device = (
                inference.create_random_from_sparse_repeat(prob.m, prob.n, prob.row_offsets.cpu(), prob.col_ptr.cpu(),
                                                      prob.values.cpu(), r, device=device))

            nnzs_sorted = spqr_device.row_offsets.diff().sort().values
            y = (nnzs_sorted.cpu().numpy() / spqr_host.n)

            x = np.arange(spqr_host.m) / spqr_host.m

            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=tensor_name))

    fig.show()


class Tiles:
    def __init__(self, p):
        t = torch.load(p, map_location='cpu')
        self.m = t['weight_shape'][0]
        self.n = t['weight_shape'][1]

        self.beta1 = t['groupsize']
        self.beta2 = t['groupsize']

        self.total_blocks = updiv(self.m, self.beta1) * updiv(self.n, self.beta2)

        outliers_matrix = t['outliers_matrix']
        outliers_matrix = outliers_matrix.to_sparse_csr()
        self.values = outliers_matrix.values().float() * 0.0 + 1.0
        self.col_ptr = outliers_matrix.col_indices().int()
        self.row_offsets = outliers_matrix.crow_indices().int()
        self.row_nnzs = self.row_offsets.diff()
        self.perc_row_empty = (self.row_nnzs == 0).sum() / self.m
        self.nnz = self.values.shape[0]

        csrmat = csr_matrix((self.values, self.col_ptr, self.row_offsets), shape=(self.m, self.n))

        densemat = csrmat.todense()

        bsr = torch.tensor(densemat, dtype=torch.float32).to_sparse_bsr(blocksize=(self.beta1, self.beta2))

        self.tile_row_offsets = bsr.crow_indices().int()
        self.tile_count = bsr.values().shape[0]
        self.tile_nnzs = bsr.values().flatten(start_dim=-2).sum(axis=1)
        self.tile_row_count = self.tile_row_offsets.shape[0] - 1



class Sparisty:
    def __init__(self, p):
        t = torch.load(p, map_location='cpu')
        self.m = t['weight_shape'][0]
        self.n = t['weight_shape'][1]
        outliers_matrix = t['outliers_matrix']
        outliers_matrix = outliers_matrix.to_sparse_csr()
        self.row_offsets = outliers_matrix.crow_indices().int()
        self.row_nnzs = self.row_offsets.diff()
        self.nnz = self.row_nnzs.sum()


def prettify_layer_name(p: str):
    return p.replace('self_attn.', '').replace('mlp.', '').replace('_proj', '').upper()


def densities_tile():
    fig = go.Figure()
    device = torch.device('cuda:0')

    base_path = sys.argv[1]
    folders = os.listdir(base_path)
    args_path = os.path.join(base_path, 'args.pt')
    args = torch.load(args_path)
    # fig.update_xaxes(type='log')

    fig.update_xaxes(type='log')

    for f in [fig]:
        f.update_layout(title='Layer-Aggregate Tile Densities',
                        title_x=0.5,
                        xaxis_title='Tile density (%) log',
                        yaxis_title='Tile count ratio',
                        margin=dict(l=0, r=0, t=30, b=0))

    folders.sort()

    x = np.arange(257)

    for l_id, layer_id in enumerate(folders):
        tile_nnzs = np.zeros(257)
        folder = os.path.join(base_path, layer_id)
        if not os.path.isdir(folder):
            continue
        layer_count = 0

        per_layer_figure = go.Figure()
        f.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        other_color = 'green'
        for p in os.listdir(folder):
            tensor_name = prettify_layer_name(p)
            per_layer_figure.update_layout(title=f'Layer {layer_id}', title_x=0.5, xaxis_title='Tile density log (%)',
                                           yaxis_title='Tile count (normalized)')
            per_layer_figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            per_layer_figure.update_xaxes(type='log')

            tiles = Tiles(os.path.join(folder, p))
            print(f'{tensor_name} perc rows empty = {tiles.perc_row_empty * 100}%')
            total_nnz_blocks = float(tiles.tile_nnzs.shape[0])
            y, _ = np.histogram(tiles.tile_nnzs, bins=range(258))
            y = y / total_nnz_blocks
            tile_nnzs += y
            layer_count += 1

            per_layer_figure.add_trace(go.Line(x=(x / 256) * 100, y=y.cumsum(), name=tensor_name))

            per_layer_figure.write_image(f'report/layers/layer{layer_id:02}.png')

        # per_layer_figure.show()
        cumsum = (tile_nnzs / layer_count).cumsum()

        if layer_id == '00' or layer_id == '01':
            fig.add_trace(go.Line(x=(x / 256) * 100, y=cumsum, name=f'Layer {layer_id}', line_color=colors[l_id]))
        elif layer_id == '02':
            fig.add_trace(go.Line(x=(x / 256) * 100, y=cumsum, name=f'Other layers', line_color=colors[2]))
        else:
            fig.add_trace(go.Line(x=(x / 256) * 100, y=cumsum, line_color=colors[2], showlegend=False))

    # fig.show()
    fig.write_image(f'report/layer_aggregate.png', scale=4)


def densities_rows_log_index():
    base_path = sys.argv[1]
    folders = os.listdir(base_path)

    folders.sort()

    for l_id, layer_id in enumerate(folders):
        folder = os.path.join(base_path, layer_id)
        if not os.path.isdir(folder):
            continue
        layer_count = 0

        per_layer_figure = go.Figure()

        for p in os.listdir(folder):
            tensor_name = prettify_layer_name(p)
            per_layer_figure.update_layout(title=f'Layer {layer_id}', title_x=0.5, xaxis_title='Row index (normalized) log',
                                           yaxis_title='Row density (normalized) log')
            per_layer_figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            # per_layer_figure.update_xaxes(type='log')
            # per_layer_figure.update_yaxes(type='log')

            tiles = Tiles(os.path.join(folder, p))
            print(f'{tensor_name} perc rows empty = {tiles.perc_row_empty * 100}%')
            y = tiles.row_nnzs.sort().values.numpy()
            y = y / tiles.n
            print(((tiles.row_nnzs.reshape((-1, 16)).sum(axis=1) / tiles.n) * 100).int().max())
            layer_count += 1

            x_axis = torch.arange(tiles.m) / tiles.m
            per_layer_figure.add_trace(go.Line(x=x_axis, y=y, name=tensor_name))

            per_layer_figure.write_image(f'report/rows/layer{layer_id:02}.png')
        per_layer_figure.show()

def densities_rows():
    base_path = sys.argv[1]
    folders = os.listdir(base_path)

    folders.sort()

    for l_id, layer_id in enumerate(folders):
        folder = os.path.join(base_path, layer_id)
        if not os.path.isdir(folder):
            continue
        layer_count = 0

        per_layer_figure = go.Figure()

        for p in os.listdir(folder):
            tensor_name = prettify_layer_name(p)
            per_layer_figure.update_layout(title=f'Layer {layer_id}', title_x=0.5, xaxis_title='Row index',
                                           yaxis_title='Row density')
            per_layer_figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            # per_layer_figure.update_xaxes(type='log')
            # per_layer_figure.update_yaxes(type='log')

            tiles = Tiles(os.path.join(folder, p))
            print(f'{tensor_name} perc rows empty = {tiles.perc_row_empty * 100}%')
            y = tiles.row_nnzs.sort().values.numpy()
            y = y / tiles.n
            print(((tiles.row_nnzs.reshape((-1, 16)).sum(axis=1) / tiles.n) * 100).int().max())
            layer_count += 1

            x_axis = torch.arange(tiles.m) / tiles.m
            per_layer_figure.add_trace(go.Line(x=x_axis, y=y, name=tensor_name))

            per_layer_figure.write_image(f'report/rows/layer{layer_id:02}.png')
        per_layer_figure.show()

def densities_rows_imbalance():
    base_path = sys.argv[1]
    folders = os.listdir(base_path)

    folders.sort()

    res = {}

    for l_id, layer_id in enumerate(folders):
        folder = os.path.join(base_path, layer_id)
        if not os.path.isdir(folder):
            continue
        layer_count = 0


        for p in os.listdir(folder):
            tensor_name = prettify_layer_name(p)

            sparsity = Sparisty(os.path.join(folder, p))
            print(f'{tensor_name}')
            nnzs = sparsity.row_nnzs.numpy()
            y_groups = nnzs.reshape((-1, 16))

            y_max = y_groups.max(axis=1)

            repeated = np.repeat(y_max, 16)
            valid = repeated != 0

            v = (repeated[valid] - nnzs[valid]).mean()

            if not tensor_name in res.keys():
                res[tensor_name] = np.array([v], dtype=np.float64)
            else:
                res[tensor_name] = np.append(res[tensor_name], v)
            layer_count += 1

    layer_ids = np.arange(res['K'].shape[0])

    per_layer_figure = go.Figure()

    for t, v in res.items():
        per_layer_figure.add_trace(go.Scatter(x=layer_ids, y=v, name=t, mode='lines'))

    per_layer_figure.update_layout(title=f'Std Devs', title_x=0.5, xaxis_title='Layer Id',
                                   yaxis_title='Std Dev (Normalized)')
    per_layer_figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    # per_layer_figure.update_xaxes(type='log')
    # per_layer_figure.update_yaxes(type='log')

    # per_layer_figure.add_trace(go.Line(x=x_axis, y=y, name=tensor_name))
    # per_layer_figure.write_image(f'report/rows_imbalance/layer{layer_id:02}.png')
    per_layer_figure.show()


if __name__ == '__main__':
    densities_rows()
