import collections as co
import typing as t

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from evaldet.tracks import Tracks

from .utils import create_coo_array

_EPS = 1 / 1000


class HOTAResults(t.TypedDict):
    HOTA: float
    DetA: float
    AssA: float
    LocA: float
    alphas_HOTA: np.ndarray
    HOTA_alpha: np.ndarray
    DetA_alpha: np.ndarray
    AssA_alpha: np.ndarray
    LocA_alpha: np.ndarray


def calculate_hota_metrics(
    ground_truth: Tracks, hypotheses: Tracks, ious: dict[int, npt.NDArray[np.float32]]
) -> HOTAResults:
    """Calculate HOTA metrics.

    Args:
        ground_truth: Ground truth tracks.
        hypotheses: Hypotheses tracks.
        ious_threshold: A dictionary where keys are frame numbers (indices), and values
            are numpy matrices of IOU distances between detection in ground truth and
            hypotheses for that frame. IOUs must be present for all frames that are
            present in both ground truth and hypotheses.

    Returns:
        A dictionary containing HOTA metrics (both average and individual alpha values).
        Note that I use the matching algorithm from the paper, which differs from what
        the official repository (TrackEval) is using - see
        [this issue](https://github.com/JonathonLuiten/TrackEval/issues/22) for more
        details. The metrics returned are:

            - HOTA
            - AssA
            - DetA
            - LocA
    """

    alphas = np.arange(0.05, 0.96, 0.05)  # from 0.05 to 0.95 inclusive
    all_frames = sorted(set(ground_truth.frames).intersection(hypotheses.frames))

    gts = tuple(ground_truth.ids_count.keys())
    gts_counts = tuple(ground_truth.ids_count.values())
    gts_id_ind_dict = {_id: ind for ind, _id in enumerate(gts)}

    hyps = tuple(hypotheses.ids_count.keys())
    hyps_counts = tuple(hypotheses.ids_count.values())
    hyps_id_ind_dict = {_id: ind for ind, _id in enumerate(hyps)}

    n_gt, n_hyp = len(gts), len(hyps)
    FP, FN = sum(hyps_counts), sum(gts_counts)

    DetAs = np.zeros_like(alphas)
    AssAs = np.zeros_like(alphas)
    LocAs = np.zeros_like(alphas)

    for a_ind, alpha in enumerate(alphas):
        # The arrays should all have the shape [n_gt, n_hyp]
        FPA_max = np.tile(hyps_counts, (n_gt, 1))
        FNA_max = np.tile(gts_counts, (n_hyp, 1)).T
        TPA_max_vals: dict[tuple[int, int], int] = co.defaultdict(int)

        FPA, FNA = FPA_max.copy(), FNA_max.copy()
        locs = 0.0  # Accumulator of similarities

        # Do the optimisitc matching - allow multiple matches per gt/hyp in the
        # same frame
        for frame in all_frames:
            dist_matrix = ious[frame]

            gt_frame_inds = [gts_id_ind_dict[_id] for _id in ground_truth[frame].ids]
            hyp_frame_inds = [hyps_id_ind_dict[_id] for _id in hypotheses[frame].ids]

            for row_ind, col_ind in np.argwhere(dist_matrix < alpha):
                TPA_max_vals[(gt_frame_inds[row_ind], hyp_frame_inds[col_ind])] += 1

        TPA_max = create_coo_array(TPA_max_vals, (n_gt, n_hyp)).toarray()

        # Compute optimistic A_max, to be used for actual matching
        A_max = TPA_max / (FNA_max + FPA_max - TPA_max)

        # Do the actual matching
        TPA_vals: dict[tuple[int, int], int] = co.defaultdict(int)
        for frame in all_frames:
            dist_matrix = ious[frame]
            dist_cost = (1 - dist_matrix) * _EPS

            gt_ids_f = ground_truth[frame].ids
            hyp_ids_f = hypotheses[frame].ids
            gt_frame_inds = [gts_id_ind_dict[_id] for _id in gt_ids_f]
            hyp_frame_inds = [hyps_id_ind_dict[_id] for _id in hyp_ids_f]

            opt_matrix = ((dist_matrix < alpha) / _EPS).astype(np.float64)
            opt_matrix += A_max[np.ix_(gt_frame_inds, hyp_frame_inds)]
            opt_matrix += dist_cost

            # Calculate matching as a LAP
            matching_inds = linear_sum_assignment(opt_matrix, maximize=True)
            for row_ind, col_ind in zip(*matching_inds):
                if dist_matrix[row_ind, col_ind] < alpha:
                    TPA_vals[(gt_frame_inds[row_ind], hyp_frame_inds[col_ind])] += 1
                    locs += 1 - dist_matrix[row_ind, col_ind]

        TPA = create_coo_array(TPA_vals, (n_gt, n_hyp)).toarray()

        # Compute proper scores
        TP = TPA.sum()
        A = TPA / (FNA + FPA - TPA)
        DetAs[a_ind] = TP / (FN + FP - TP)
        AssAs[a_ind] = (TPA * A).sum() / max(TP, 1)

        # If no matches -> full similarity [strange default]
        LocAs[a_ind] = np.maximum(locs, 1e-10) / np.maximum(TP, 1e-10)

    HOTAs = np.sqrt(DetAs * AssAs)

    return {
        "HOTA": HOTAs.mean(),
        "DetA": DetAs.mean(),
        "AssA": AssAs.mean(),
        "LocA": LocAs.mean(),
        "alphas_HOTA": alphas,
        "HOTA_alpha": HOTAs,
        "DetA_alpha": DetAs,
        "AssA_alpha": AssAs,
        "LocA_alpha": LocAs,
    }
