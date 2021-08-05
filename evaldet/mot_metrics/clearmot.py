import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dist import iou_dist, iou_dist_pairwise
from ..tracks import Tracks

logger = logging.getLogger(__name__)


def _indices_present(big_list: List, candidates: List) -> List[int]:
    return [big_list.index(i) for i in candidates]


def _indices_non_present(big_list: List, candidates: List) -> List[int]:
    return [i for i, val in enumerate(big_list) if val not in candidates]


def _get_matching_detections(
    matching: Dict[int, int], gt: Dict[str, Any], hyp: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    ind_matching_gt = _indices_present(gt["ids"], list(matching.keys()))
    ind_matching_hyp = _indices_present(hyp["ids"], list(matching.values()))

    matching_det_gt = gt["detections"][ind_matching_gt]
    matching_det_hyp = hyp["detections"][ind_matching_hyp]

    return matching_det_gt, matching_det_hyp


def _get_non_matching_detections(
    matching: Dict[int, int], gt: Dict[str, Any], hyp: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    ind_matching_gt = _indices_non_present(gt["ids"], list(matching.keys()))
    ind_matching_hyp = _indices_non_present(hyp["ids"], list(matching.values()))

    matching_det_gt = gt["detections"][ind_matching_gt]
    matching_det_hyp = hyp["detections"][ind_matching_hyp]

    return matching_det_gt, matching_det_hyp


def calculate_clearmot_metrics(
    ground_truth: Tracks, hypotheses: Tracks, dist_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:

    all_frames = sorted(set(ground_truth.frames + hypotheses.frames))

    false_negatives = 0
    false_positives = 0
    mismatches = 0
    ground_truths = 0

    matches_dist = []
    matching: Dict[int, int] = {}

    for frame in all_frames:
        if frame not in ground_truth:
            matching = {}
            false_positives += len(hypotheses[frame]["ids"])
        elif frame not in hypotheses:
            matching = {}
            false_negatives += len(ground_truth[frame]["ids"])
            ground_truths += len(ground_truth[frame]["ids"])
        else:
            old_matching = matching.copy()
            hyp, gt = hypotheses[frame], ground_truth[frame]
            ground_truths += len(ground_truth[frame]["ids"])

            # Delete from matching missing gts/detections
            for missing_key in set(matching.keys()) - set(gt["ids"]):
                del matching[missing_key]

            for key in list(matching.keys()):
                if matching[key] not in hyp["ids"]:
                    del matching[key]

            # For remaining matching, check that dist below threshold
            matching_det_gt, matching_det_hyp = _get_matching_detections(
                matching, gt, hyp
            )
            correspond_dists = iou_dist_pairwise(matching_det_gt, matching_det_hyp)
            for key, dist in zip(list(matching.keys()), correspond_dists):
                if dist > dist_threshold or np.isnan(dist):
                    del matching[key]
                else:
                    matches_dist.append(dist)

            # For remaining gts/detections, compute matching as a LAP
            ids_non_matching_gt = [
                _id for _id in gt["ids"] if _id not in matching.keys()
            ]
            ids_non_matching_hyp = [
                _id for _id in hyp["ids"] if _id not in matching.values()
            ]
            non_matching_det_gt, non_matching_det_hyp = _get_non_matching_detections(
                matching, gt, hyp
            )

            dist_matrix = iou_dist(non_matching_det_gt, non_matching_det_hyp)
            row_inds, col_inds = linear_sum_assignment(dist_matrix)
            for row_ind, col_ind in zip(row_inds, col_inds):
                if dist_matrix[row_ind, col_ind] < dist_threshold:
                    matching[ids_non_matching_gt[row_ind]] = ids_non_matching_hyp[
                        col_ind
                    ]
                    matches_dist.append(dist_matrix[row_ind, col_ind])

            # Check if mismatches have occured
            for key in matching:
                if key in old_matching and matching[key] != old_matching[key]:
                    mismatches += 1

            # Compute false positives and false negatices
            false_negatives += len(gt["ids"]) - len(matching)
            false_positives += len(hyp["ids"]) - len(matching)

    if not matches_dist:
        logger.warning("No matches were made, MOPT will be np.nan")
        motp = np.nan
    else:
        motp = sum(matches_dist) / len(matches_dist)

    mota = 1 - (false_negatives + false_positives + mismatches) / ground_truths

    return {"motp": motp, "mota": mota}
