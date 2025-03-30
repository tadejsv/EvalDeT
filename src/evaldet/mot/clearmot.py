"""CLEARMOT MOT metrics."""

import logging
import typing as t

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from evaldet.tracks import FrameTracks, Tracks

logger = logging.getLogger(__name__)


class CLEARMOTResults(t.TypedDict):
    """
    A typed dictionary for storing the results of the CLEARMOT metric evaluation.
    """

    MOTP: float
    MOTA: float
    FP_CLEAR: float
    FN_CLEAR: float
    IDSW: float


def _update_frame_matches(
    matching: dict[int, int],
    matching_persist: dict[int, int],
    gt: FrameTracks,
    hyp: FrameTracks,
    iou_frame: npt.NDArray[np.float32],
    dist_threshold: float,
    matches_dist: list[float],
    mismatches: int,
) -> int:
    """
    Update `matching` for the current frame based on IOU distances and
    return updated `mismatches`.
    """
    # Remove matches that do not exist in this frame's GT or Hyp
    for key in list(matching.keys()):
        if matching[key] not in hyp.ids or key not in gt.ids:
            del matching[key]

    # Build index dictionaries for GT and hypotheses
    id_ind_dict_gt = {_id: ind for ind, _id in enumerate(gt.ids)}
    id_ind_dict_hyp = {_id: ind for ind, _id in enumerate(hyp.ids)}

    # Remove matches above the distance threshold; store valid matches' distances
    for gt_id, hyp_id in tuple(matching.items()):
        dist_match = iou_frame[id_ind_dict_gt[gt_id], id_ind_dict_hyp[hyp_id]]
        if dist_match > dist_threshold:
            del matching[gt_id]
        else:
            matches_dist.append(dist_match)

    # Identify which GT and Hyp IDs remain unmatched
    match_gt_ids = tuple(matching.keys())
    match_hyp_ids = tuple(matching.values())
    inds_nm_gt = np.where(np.isin(gt.ids, match_gt_ids, invert=True))[0]
    inds_nm_hyp = np.where(np.isin(hyp.ids, match_hyp_ids, invert=True))[0]
    dist_matrix = iou_frame[np.ix_(inds_nm_gt, inds_nm_hyp)]

    # Assign remaining unmatched IDs using LAP
    for row_ind, col_ind in zip(*linear_sum_assignment(dist_matrix), strict=True):
        if dist_matrix[row_ind, col_ind] < dist_threshold:
            matching[gt.ids[inds_nm_gt[row_ind]]] = hyp.ids[inds_nm_hyp[col_ind]]
            matches_dist.append(dist_matrix[row_ind, col_ind])

    # Check if mismatches (ID switches) have occurred
    for key, matched_track in matching.items():
        if key in matching_persist and matched_track != matching_persist[key]:
            mismatches += 1

    return mismatches


def calculate_clearmot_metrics(
    ground_truth: Tracks,
    hypotheses: Tracks,
    ious: dict[int, npt.NDArray[np.float32]],
    dist_threshold: float = 0.5,
) -> CLEARMOTResults:
    """
    Calculate CLEARMOT metrics.

    Args:
        ground_truth: Ground truth tracks.
        hypotheses: Hypotheses tracks.
        ious: A dictionary where keys are frame numbers (indices), and values
            are numpy matrices of IOU distances between detection in ground truth and
            hypotheses for that frame. IOUs must be present for all frames that are
            present in both ground truth and hypotheses.
        dist_threshold: The distance threshold for the computation of metrics, used to
            determine whether a matching between two tracks persist, and whether to
            start a matching based on distance between two detections.

    Returns:
        A dictionary containing CLEARMOT metrics:

            - MOTA (MOT Accuracy)
            - MOTP (MOT Precision)
            - FP (false positives)
            - FN (false negatives)
            - IDS (identity switches/mismatches)

    """
    all_frames = sorted(ground_truth.frames.union(hypotheses.frames))

    false_negatives = 0
    false_positives = 0
    mismatches = 0
    ground_truths = 0

    matches_dist: list[float] = []
    matching: dict[int, int] = {}

    # This is the persistent matching dictionary, used to check for mismatches
    # when a previously matched hypothesis is re-matched with a ground truth
    matching_persist: dict[int, int] = {}

    for frame in all_frames:
        if frame not in ground_truth:
            matching = {}
            false_positives += len(hypotheses[frame].ids)
        elif frame not in hypotheses:
            matching = {}
            false_negatives += len(ground_truth[frame].ids)
            ground_truths += len(ground_truth[frame].ids)
        else:
            hyp, gt = hypotheses[frame], ground_truth[frame]

            mismatches = _update_frame_matches(
                matching=matching,
                matching_persist=matching_persist,
                gt=gt,
                hyp=hyp,
                iou_frame=ious[frame],
                dist_threshold=dist_threshold,
                matches_dist=matches_dist,
                mismatches=mismatches,
            )

            # Compute false positives and false negatives
            false_negatives += len(gt.ids) - len(matching)
            false_positives += len(hyp.ids) - len(matching)

            ground_truths += len(gt.ids)
            matching_persist.update(matching)

    if not matches_dist:
        logger.warning("No matches were made, MOPT will be np.nan")
        motp = np.nan
    else:
        motp = sum(matches_dist) / len(matches_dist)

    mota = 1 - (false_negatives + false_positives + mismatches) / ground_truths

    return {
        "MOTP": motp,
        "MOTA": mota,
        "FP_CLEAR": false_positives,
        "FN_CLEAR": false_negatives,
        "IDSW": mismatches,
    }
