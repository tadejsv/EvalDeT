import logging
from typing import Dict, Union

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore

from ..dist import iou_dist
from ..tracks import Tracks

logger = logging.getLogger(__name__)


def calculate_id_metrics(
    ground_truth: Tracks, hypotheses: Tracks, dist_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    pass

    gts = tuple(ground_truth.ids_count.keys())
    hyps = tuple(hypotheses.ids_count.keys())
    n_gt, n_hyp = len(gts), len(hyps)
    matching = np.zeros((n_gt, n_hyp), dtype=np.int32)

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

    fn_matrix = np.tile(list(ground_truth.ids_count.values()), (n_hyp, 1)).T
    fp_matrix = np.tile(list(hypotheses.ids_count.values()), (n_gt, 1))
    cost_matrix = fp_matrix + fn_matrix - 2 * matching

    # Calculate matching as a LAP, get FN and FP from matched entries
    remaining_gts, remaining_hyps = list(range(n_gt)), list(range(n_hyp))
    false_positive, false_negative, true_positive = 0, 0, 0
    for row_ind, col_ind in zip(*linear_sum_assignment(cost_matrix)):
        false_negative += fn_matrix[row_ind, 0] - matching[row_ind, col_ind]
        false_positive += fp_matrix[0, col_ind] - matching[row_ind, col_ind]
        true_positive += matching[row_ind, col_ind]
        remaining_gts.remove(row_ind)
        remaining_hyps.remove(col_ind)

    # Add false negatives / positives for unmatches indices
    for non_matched_gt in remaining_gts:
        false_negative += fn_matrix[non_matched_gt, 0]
    for non_matched_hyp in remaining_hyps:
        false_positive += fp_matrix[0, non_matched_hyp]

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
