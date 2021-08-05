import logging
from typing import Dict, Union

import numpy as np

from ..dist import iou_dist, iou_dist_pairwise
from ..tracks import Tracks

logger = logging.getLogger(__name__)


def calculate_clearmot_metrics(
    ground_truth: Tracks, detections: Tracks, dist_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:

    all_frames = sorted(set(ground_truth.frames + detections.frames))

    false_negatives = 0
    false_positives = 0
    mismatches = 0
    ground_truths = 0

    matches_dist = []
    matching = {}

    for frame in all_frames:
        if frame not in ground_truth:
            matching = {}
            false_positives += len(detections[frame])
        elif frame not in detections:
            matching = {}
            false_negatives += len(ground_truth[frame])
            ground_truths += len(ground_truth[frame])
        else:
            old_matching = matching.copy()

            # Delete from matching missing gts/detections
            for missing_key in set(matching.keys()) - set(ground_truth[frame]["ids"]):
                del matching[missing_key]

            for key in list(matching.keys()):
                if matching[key] not in detections[frame]["ids"]:
                    del matching[key]

            # For remaining matching, check that dist below threshold
            matching_gt = []
            matching_det = []
            correspond_dists = iou_dist_pairwise(matching_gt, matching_det)
            for key, dist in zip(list(matching.keys()), correspond_dists):
                if dist > dist_threshold or np.isnan(dist):
                    del matching[key]

            # For remaining gts/detections, compute matching as a LAP

            # Check if mismatches have occured

            # Compute false positives and false negatices

    if not matches_dist:
        logger.warning("No matches were made, MOPT will be np.nan")
        motp = np.nan
    else:
        motp = sum(matches_dist) / len(matches_dist)

    mota = 1 - (false_negatives + false_positives + mismatches) / ground_truths

    return {"motp": motp, "mota": mota}
