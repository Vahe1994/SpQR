from typing import Tuple

import torch
from torch import Tensor as T


def merge_col_val(col_ids, values) -> T:
    """
    Merge 16-bit col ids buffer with the 16-bi values buffer into a single buffer with the columns
    occupying the lower half of the 32-bit number.

    @param col_ids: CSR column ids
    @param values:  CSR values
    @return: Merged colvals buffer.
    """
    if values.shape[0] != 0:
        return (
            values.view(torch.int16)
            .to(torch.int64)
            .bitwise_left_shift(16)
            .bitwise_or(col_ids.view(torch.int16).to(torch.int64))
            .to(torch.int32)
        )
    else:
        return torch.zeros(0)


def init_ptcsr(row_offsets) -> Tuple[T, T]:
    """
    Given a row_offsets buffer in CSR, produce a valid PTCSR row_offsets buffer and
    allocate the PTCSR col_vals buffer. Later on we pass these buffers down into a
    C++ conversion implementation which does not do any extra allocations. The row
    offsets calculation is simple enough to be done in a vectorized form.

    @param row_offsets: CSR row offsets
    @return A tuple of PTCSR Row offsets and an allocate PTCSR col_vals buffer.
    """
    row_offsets_output = row_offsets.diff().reshape((-1, 16)).max(axis=1).values * 16
    row_offsets_output = row_offsets_output.cumsum(dim=0)
    row_offsets_output = torch.cat((torch.tensor([0]), row_offsets_output)).to(dtype=torch.int32)
    col_val_count = row_offsets_output[-1]
    col_vals_output = torch.zeros(col_val_count, dtype=torch.int32)
    return row_offsets_output, col_vals_output
