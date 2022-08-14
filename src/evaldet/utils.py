import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore

from . import Tracks
from .dist import iou_dist


def preprocess_mot_1720(gt: Tracks, hyp: Tracks, mot_20: bool = True):
    distractor_cls_ids = [2, 7, 8, 12]
    if mot_20:
        distractor_cls_ids.append(6)

    pedestrian_class = [1]

    all_frames = sorted(set(gt.frames).intersection(hyp.frames))
    for frame in all_frames:
        dist_matrix = iou_dist(gt[frame]["detections"], hyp[frame]["detections"])
        dist_matrix[dist_matrix > 0.5] = 1
        matches_row, matches_col = linear_sum_assignment(dist_matrix)

        matches_filter = dist_matrix[matches_row, matches_col] < 0.5
        matches_row = matches_row[matches_filter]
        matches_col = matches_col[matches_filter]

        is_distractor = np.in1d(
            gt[frame]["classes"][matches_row], distractor_cls_ids, assume_unique=True
        )
        to_remove_hyp = np.in1d(
            np.arange(len(hyp[frame]["ids"])),
            matches_col[is_distractor],
            assume_unique=True,
        )

        hyp.filter_frame(frame, ~to_remove_hyp)

    gt.filter_by_conf(0.1)
    gt.filter_by_class(pedestrian_class)
