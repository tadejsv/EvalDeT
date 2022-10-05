import typing as t

import numpy as np
from scipy import sparse


def create_coo_array(
    vals_dict: t.Dict[t.Tuple[int, int], int], shape: t.Tuple[int, int]
) -> sparse.coo_array:
    """Create a sparse COO array.

    Args:
        vals_dict: A dictionary with values. The key should be a tuple of
            ``(row_ind, col_ind)``, and the value should be the entry for the cell
            at that index.
        shape: Shape of the new array: ``(n_rows, n_cols)``
    """
    row_inds = np.array(tuple(x[0] for x in vals_dict.keys()))
    col_inds = np.array(tuple(x[1] for x in vals_dict.keys()))
    vals = np.array(tuple(vals_dict.values()))

    return sparse.coo_array((vals, (row_inds, col_inds)), shape=shape)
