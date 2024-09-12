import numpy as np
import torch
import time
import os
import sys

from scipy.sparse import csr_matrix, coo_matrix

import inference
from problem import Problem

import plotly.graph_objects as go
import plotly.express as px

colors = px.colors.sequential.Plasma
num_colors = len(colors)
colors = [colors[0], colors[num_colors // 2 - 1], colors[-4]]


def updiv(x, y): return (x + y - 1) // y


def densities():
    fig = go.Figure()
    device = torch.device('cuda:0')

    base_path = sys.argv[1]
    folders = os.listdir(base_path)
    args_path = os.path.join(base_path, 'args.pt')
    args = ProblemArgs().load_args(args_path)

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
            tensor_name = f'L{layer_id} {p}'
            prob = Problem(os.path.join(folder, p), args, f'{layer_id}_{p}', device=device)

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
            per_layer_figure.update_layout(title=f'Layer {layer_id}', title_x=0.5, xaxis_title='Row index (normalized) log',
                                           yaxis_title='Row density (normalized) log')
            per_layer_figure.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            per_layer_figure.update_xaxes(type='log')
            per_layer_figure.update_yaxes(type='log')

            tiles = Tiles(os.path.join(folder, p))
            print(f'{tensor_name} perc rows empty = {tiles.perc_row_empty * 100}%')
            y = tiles.row_nnzs.sort().values.numpy()
            y = y / tiles.n
            layer_count += 1

            x_axis = torch.arange(tiles.m) / tiles.m
            per_layer_figure.add_trace(go.Line(x=x_axis, y=y, name=tensor_name))

            per_layer_figure.write_image(f'report/rows/layer{layer_id:02}.png')
        per_layer_figure.show()


if __name__ == '__main__':
    densities_rows()
