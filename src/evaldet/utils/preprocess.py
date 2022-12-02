import numpy as np
from scipy.optimize import linear_sum_assignment

from evaldet import Tracks
from evaldet.dist import iou_dist


def preprocess_mot_1720(gt: Tracks, hyp: Tracks, mot_20: bool = True) -> None:
    distractor_cls_ids = [2, 7, 8, 12]
    if mot_20:
        distractor_cls_ids.append(6)

    pedestrian_class = [1]

    hyp_filter = np.ones_like(hyp.ids, dtype=np.bool_)
    gt_filter = np.ones_like(gt.ids, dtype=np.bool_)

    all_frames = sorted(set(gt.frames).intersection(hyp.frames))
    for frame in all_frames:
        dist_matrix = iou_dist(gt[frame].detections, hyp[frame].detections)
        dist_matrix[dist_matrix > 0.5] = 1
        matches_row, matches_col = linear_sum_assignment(dist_matrix)

        matches_filter = dist_matrix[matches_row, matches_col] < 0.5
        matches_row = matches_row[matches_filter]
        matches_col = matches_col[matches_filter]

        classes = gt[frame].classes
        assert classes is not None
        is_distractor = np.in1d(
            classes[matches_row], distractor_cls_ids, assume_unique=True
        )
        to_remove_hyp = np.in1d(
            np.arange(len(hyp[frame].ids)),
            matches_col[is_distractor],
            assume_unique=True,
        )

        start, end = hyp._frame_ind_dict[frame]
        hyp_filter[start:end] = ~to_remove_hyp

    gt_filter = gt_filter & (gt.confs >= 0.1)
    gt_filter = gt_filter & np.isin(gt.classes, pedestrian_class)

    gt.filter(gt_filter)
    hyp.filter(hyp_filter)
