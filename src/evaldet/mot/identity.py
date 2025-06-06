"""ID family of MOT metrics."""

import collections as co
import typing as t

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from evaldet.tracks import Tracks

from .utils import create_coo_array


class IDResults(t.TypedDict):
    """
    A typed dictionary for storing the results of the ID metric evaluation.
    """

    IDTP: float
    IDFP: float
    IDFN: float
    IDP: float
    IDR: float
    IDF1: float


def calculate_id_metrics(
    ground_truth: Tracks,
    hypotheses: Tracks,
    ious: dict[int, npt.NDArray[np.float32]],
    dist_threshold: float = 0.5,
) -> IDResults:
    """
    Calculate ID (identity) metrics.

    Args:
        ground_truth: Ground truth tracks.
        hypotheses: Hypotheses tracks.
        ious: A dictionary where keys are frame numbers (indices), and values
            are numpy matrices of IOU distances between detection in ground truth and
            hypotheses for that frame. IOUs must be present for all frames that are
            present in both ground truth and hypotheses.
        dist_threshold: The distance threshold for the computation of the metrics - used
            to determine whether to match two objects.

    Returns:
        A dictionary containing ID metrics:

            - IDP (ID Precision)
            - IDR (ID Recall)
            - IDF1 (ID F1)
            - IDFP (ID false positives)
            - IDFN (ID false negatives)
            - IDTP (ID true positives)

    """
    gts = tuple(ground_truth.ids_count.keys())
    gts_id_ind_dict = {_id: ind for ind, _id in enumerate(gts)}
    hyps = tuple(hypotheses.ids_count.keys())
    hyps_id_ind_dict = {_id: ind for ind, _id in enumerate(hyps)}

    gts_counts = np.array(tuple(ground_truth.ids_count.values()), dtype=np.int32)
    hyps_counts = np.array(tuple(hypotheses.ids_count.values()), dtype=np.int32)

    matches: dict[tuple[int, int], int] = co.defaultdict(int)
    for frame in sorted(set(ground_truth.frames).intersection(hypotheses.frames)):
        dist_matrix = ious[frame]
        gt_frame_inds = [gts_id_ind_dict[_id] for _id in ground_truth[frame].ids]
        htp_frame_inds = [hyps_id_ind_dict[_id] for _id in hypotheses[frame].ids]

        for gt_ind, hyp_ind in np.argwhere(dist_matrix < dist_threshold):
            matches[(gt_frame_inds[gt_ind], htp_frame_inds[hyp_ind])] += 1

    # Calculate matching as a LAP, get FN, FP and TP from matched entries
    matches_matrix = create_coo_array(matches, (len(gts), len(hyps)))
    matches_array = matches_matrix.toarray()
    row_m_inds, col_m_inds = linear_sum_assignment(matches_array, maximize=True)

    # Calculate the final results
    TPs = matches_array[row_m_inds, col_m_inds].sum()
    FNs = gts_counts.sum() - TPs
    FPs = hyps_counts.sum() - TPs

    idp = TPs / (TPs + FPs)
    idr = TPs / (TPs + FNs)
    idf1 = 2 * TPs / (2 * TPs + FPs + FNs)

    return {
        "IDTP": TPs,
        "IDFP": FPs,
        "IDFN": FNs,
        "IDP": idp,
        "IDR": idr,
        "IDF1": idf1,
    }
