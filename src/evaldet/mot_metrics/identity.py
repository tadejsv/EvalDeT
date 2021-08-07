from typing import Dict, Union

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore

from ..dist import iou_dist
from ..tracks import Tracks


def calculate_id_metrics(
    ground_truth: Tracks, hypotheses: Tracks, dist_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    pass

    gts = tuple(ground_truth.ids_count.keys())
    gts_counts = tuple(ground_truth.ids_count.values())
    hyps = tuple(hypotheses.ids_count.keys())
    hyps_counts = tuple(hypotheses.ids_count.values())
    n_gt, n_hyp = len(gts), len(hyps)

    # The "real" shape is [n_gt, n_hyp], the rest is for fictional
    # entries that are needed for FP and FN matrix to make the
    # LAP problem minimize the actual loss, including for unmatched entries
    matching = np.zeros((max(n_gt, n_hyp), max(n_gt, n_hyp)), dtype=np.int32)

    for frame in sorted(set(ground_truth.frames + hypotheses.frames)):
        if frame not in ground_truth or frame not in hypotheses:
            continue

        dist_matrix = iou_dist(
            ground_truth[frame]["detections"], hypotheses[frame]["detections"]
        )
        for gt_ind, hyp_ind in np.argwhere(dist_matrix < dist_threshold):
            matching[
                gts.index(ground_truth[frame]["ids"][gt_ind]),
                hyps.index(hypotheses[frame]["ids"][hyp_ind]),
            ] += 1

    fn_matrix, fp_matrix = np.zeros_like(matching), np.zeros_like(matching)
    fp_matrix[:, :n_hyp] = np.tile(hyps_counts, (max(n_hyp, n_gt), 1))
    fn_matrix[:n_gt, :] = np.tile(gts_counts, (max(n_hyp, n_gt), 1)).T

    cost_matrix = fp_matrix + fn_matrix - 2 * matching

    # Calculate matching as a LAP, get FN, FP and TP from matched entries
    matching_inds = linear_sum_assignment(cost_matrix)
    true_positive = matching[matching_inds].sum()
    false_negative = fn_matrix[matching_inds].sum() - true_positive
    false_positive = fp_matrix[matching_inds].sum() - true_positive

    # Calculate the final results
    idp = true_positive / (true_positive + false_positive)
    idr = true_positive / (true_positive + false_negative)
    idf1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

    return {
        "IDTP": true_positive,
        "IDFP": false_positive,
        "IDFN": false_negative,
        "IDP": idp,
        "IDR": idr,
        "IDF1": idf1,
    }
