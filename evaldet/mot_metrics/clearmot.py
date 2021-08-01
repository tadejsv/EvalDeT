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
    correspondence = {}

    for frame in all_frames:
        if frame not in ground_truth:
            correspondence = {}
            false_positives += len(detections[frame])
        elif frame not in detections:
            correspondence = {}
            false_negatives += len(ground_truth[frame])
            ground_truths += len(ground_truth[frame])
        else:
            # Delete from correspondence missing gt/detections
            for missing_key in set(correspondence.keys()) - set(ground_truth[frame]):
                pass

            # For remaining correspondence, chech that dist below threshold
            correspond_gt = []
            correspond_det = []
            correspond_dists = iou_dist_pairwise(correspond_gt, correspond_det)
            for key, dist in zip(list(correspondence.keys(), correspond_dists)):
                if dist > dist_threshold or np.isnan(dist):
                    del correspondence[key]

    if not matches_dist:
        logger.warning("No matches were made, MOPT will be np.nan")
        motp = np.nan
    else:
        motp = sum(matches_dist) / len(matches_dist)

    mota = 1 - (false_negatives + false_negatives + mismatches) / ground_truths

    return {"motp": motp, "mota": mota}
