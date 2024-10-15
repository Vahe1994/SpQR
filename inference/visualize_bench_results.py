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


import pandas as pd

import pandas as pd

import pandas as pd

import pandas as pd

import pandas as pd

import pandas as pd
import pyperclip

import pandas as pd
import pyperclip

import pandas as pd
import pyperclip


def escape_latex_special_chars(text):
    special_chars = {
        '%': '\\%',
        '&': '\\&',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde',
        '^': '\\textasciicircum',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


def generate_latex_booktabs_grouped_table_vertical(data, caption=None, label=None):
    latex_str = "\\begin{table}[H]\n\\centering\n"
    if caption:
        latex_str += f"\\caption{{{escape_latex_special_chars(caption)}}}\n"
    if label:
        latex_str += f"\\label{{{escape_latex_special_chars(label)}}}\n"

    layers = data[data.columns[0]].unique()

    # Define the number of columns based on the dataframe shape
    num_columns = len(data.columns[1:])
    latex_str += f"\\setlength{{\\tabcolsep}}{{4pt}} % Adjust column separation\n"
    latex_str += f"\\begin{{tabular}}{{@{{}}{'c' * num_columns}@{{}}}}\n\\toprule\n"

    # Add header row with escaped special characters
    header = " & ".join(escape_latex_special_chars(col) for col in data.columns[1:])
    latex_str += f"{header} \\\\\n"
    latex_str += "\\midrule\n"

    for layer in layers:
        group = data[data[data.columns[0]] == layer]
        latex_str += f"\\multicolumn{{{num_columns}}}{{c}}{{\\textbf{{Layer {layer}}}}} \\\\\n"
        latex_str += "\\midrule\n"

        for _, row in group.iterrows():
            # Create a list of values for the current row, escaping them
            escaped_row = [escape_latex_special_chars(str(value)) for value in row.values]
            latex_str += " & ".join(escaped_row[1:]) + " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    pyperclip.copy(latex_str)
    return latex_str


def dump_table(cols, t):
    t = t.copy(deep=True)

    t['Size'] = '$' + t['M'].astype(str) + ' \\times ' + t['N'].astype(str) + '$'

    t = t.sort_values(by=[cols[0], cols[1]])

    t['Tensor'] = t['Tensor Name'].map(prettify_tensor_name)
    t['Layer'] = t['Layer'].astype(str)

    t = t.drop(['M', 'N', 'Tensor Name'], axis=1)


    print(generate_latex_booktabs_grouped_table_vertical(t))

    return t


if __name__ == '__main__':
    bar_width = 0.25
    csv_path = sys.argv[1]
    gpu_name = sys.argv[2]
    results = pd.read_csv(csv_path, delimiter=';', index_col=False)

    cols = results.keys()
    group_labels = cols.to_list()[5:]
    torch_key = group_labels[0]


    speedup = (results[torch_key] / results[group_labels[1]]).to_numpy()
    print(f'Geomean speed-up = {gmean(speedup)}X')

    # speedup = (results[torch_key] / results[group_labels[2]]).to_numpy()
    # print(f'Geomean speed-up (dense only) = {gmean(speedup)}X')

    labels = results['Layer'].astype('str') + ' ' + results['Tensor Name'].map(prettify_tensor_name)

    bar_scale = 0.8

    torch_results = results[torch_key]

    num_tests = torch_results.shape[0]

    width = 0.40
    x = np.arange(num_tests)
    num_groups = len(group_labels)


    fig = go.Figure()



    data = dump_table(cols, results)





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
    fig.update_layout(barmode='group', title=f'Results ({gpu_name})', title_x=0.5, xaxis_tickangle=-45,
                      xaxis_title='Tensor Name',
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
