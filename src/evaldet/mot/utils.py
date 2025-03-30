"""Various utils for computing MOT metrics."""

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from evaldet import Tracks
from evaldet.dist import iou_dist


def create_coo_array(
    vals_dict: dict[tuple[int, int], int], shape: tuple[int, int]
) -> sparse.coo_array:
    """
    Create a sparse COO array.

    Args:
        vals_dict: A dictionary with values. The key should be a tuple of
            ``(row_ind, col_ind)``, and the value should be the entry for the cell
            at that index.
        shape: Shape of the new array: ``(n_rows, n_cols)``

    """
    row_inds = np.array(tuple(x[0] for x in vals_dict))
    col_inds = np.array(tuple(x[1] for x in vals_dict))
    vals = np.array(tuple(vals_dict.values()))

    return sparse.coo_array((vals, (row_inds, col_inds)), shape=shape)


def preprocess_mot_1720(
    gt: Tracks, hyp: Tracks, mot_20: bool = True
) -> tuple[Tracks, Tracks]:
    """A utility for pre-processing MOT 17 and 20 files. Used in testing."""
    distractor_cls_ids = [2, 7, 8, 12]
    if mot_20:
        distractor_cls_ids.append(6)

    pedestrian_class = [1]

    hyp_filter = np.ones_like(hyp.ids, dtype=np.bool_)
    gt_filter = np.ones_like(gt.ids, dtype=np.bool_)

    all_frames = sorted(set(gt.frames).intersection(hyp.frames))
    for frame in all_frames:
        dist_matrix = iou_dist(gt[frame].bboxes, hyp[frame].bboxes)
        dist_matrix[dist_matrix > 0.5] = 1  # noqa: PLR2004
        matches_row, matches_col = linear_sum_assignment(dist_matrix)

        matches_filter = dist_matrix[matches_row, matches_col] < 0.5  # noqa: PLR2004
        matches_row = matches_row[matches_filter]
        matches_col = matches_col[matches_filter]

        classes = gt[frame].classes

        is_distractor = np.isin(
            classes[matches_row], distractor_cls_ids, assume_unique=True
        )
        to_remove_hyp = np.isin(
            np.arange(len(hyp[frame].ids)),
            matches_col[is_distractor],
            assume_unique=True,
        )

        start, end = hyp._frame_ind_dict[frame]  # noqa: SLF001
        hyp_filter[start:end] = ~to_remove_hyp

    gt_filter = gt_filter & (gt.confs >= 0.1)  # noqa: PLR2004
    gt_filter = gt_filter & np.isin(gt.classes, pedestrian_class)

    gt = gt.filter(gt_filter)
    hyp = hyp.filter(hyp_filter)

    return gt, hyp
