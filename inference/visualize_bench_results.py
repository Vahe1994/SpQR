import sys
from enum import IntEnum


import torch
import inference
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import gmean

def prettify_tensor_name(p: str):
    p = p.replace('self_attn.', '').replace('mlp.', '').replace('_proj', '').replace('.pth', '')
    p = f'{p.upper()[0]}{p[1:]}'
    return p


if __name__ == '__main__':
    bar_width = 0.25
    # csv_path = 'report/results_rtx_4060.csv'
    csv_path = sys.argv[1]
    results = pd.read_csv(csv_path, delimiter=';', index_col=False)

    cols = results.keys()

    group_labels = [
        # FeatureFlag.SPARSE_SHARED_BASELINE_FP16.pretty() + ' (ms)',
        inference.FeatureFlag.TORCH_FP16.pretty() + ' (ms)',
        inference.FeatureFlag.SPARSE_MIXTURE_FP32.pretty() + ' (ms)'
    ]
    torch_key = group_labels[0]

    speedup = (results[torch_key] / results[group_labels[1]]).to_numpy()
    print(f'Geomean speed-up = {gmean(speedup)}X')

    # speedup = (results[torch_key] / results[group_labels[2]]).to_numpy()
    # print(f'Geomean speed-up (dense only) = {gmean(speedup)}X')

    labels = results['Layer'] + ' ' + results['Tensor Name'].map(prettify_tensor_name)

    bar_scale = 0.8

    torch_results = results[torch_key]

    num_tests = torch_results.shape[0]

    width = 0.40
    x = np.arange(num_tests)
    num_groups = len(group_labels)

    fig = go.Figure()
    for i, g in enumerate(group_labels):
        fig.add_trace(go.Bar(
            x=labels,
            y=results[g],
            name=g,
            text=results[g],
            textposition='auto'
        ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', title='Results', title_x=0.5, xaxis_tickangle=-45, xaxis_title='Tensor Name',
                      yaxis_title='Duration (ms)')
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(0, 0, 0, 0)'))

    fig.update_layout(yaxis_range=[0, 0.5])

    if False:
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
    fig.write_image('report/bench_rtx4060_baseline.svg')
    fig.show()
