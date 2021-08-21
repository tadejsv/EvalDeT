import numpy as np
from scipy.optimize import linear_sum_assignment

from evaldet import Tracks, compute_mot_metrics
from evaldet.dist import iou_dist


_TUD_METRICS = (
    "MOTA",
    "MOTP",
    "FP_CLEAR",
    "FN_CLEAR",
    "IDS",
    "IDP",
    "IDR",
    "IDF1",
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
)


def preprocess_mot_1720(gt: Tracks, hyp: Tracks):
    distractor_cls_ids = [2, 7, 8, 12]
    pedestrian_class = [1]

    all_frames = sorted(set(gt.frames).intersection(hyp.frames))
    for frame in all_frames:
        dist_matrix = iou_dist(gt[frame]["detections"], hyp[frame]["detections"])
        dist_matrix[dist_matrix > 0.5] = 1
        matches_row, matches_col = linear_sum_assignment(dist_matrix)

        matches_filter = dist_matrix[matches_row, matches_col] < 0.5
        matches_row = matches_row[matches_filter]
        matches_col = matches_col[matches_filter]

        is_distractor = np.in1d(gt[frame]["classes"][matches_row], distractor_cls_ids)
        to_remove_hyp = np.in1d(
            np.arange(len(hyp[frame]["ids"])), matches_col[is_distractor]
        )

        hyp.filter_frame(frame, ~to_remove_hyp)

    gt.filter_by_conf(0.1)
    gt.filter_by_class(pedestrian_class)


def test_mot20_01():
    gt = Tracks.from_mot_gt("tests/data/integration/MOT20-01_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/MOT20-01_MPNTrack_hyp.csv")

    print(len(gt), len(hyp))
    # Filter to pedestrians only, and visible
    preprocess_mot_1720(gt, hyp)
    print(len(gt), len(hyp))

    results = compute_mot_metrics(metrics=_TUD_METRICS, ground_truth=gt, detections=hyp)
    exp_results = {
        "MOTA": 0.659,
        "MOTP": 1 - 0.833,
        "FP_CLEAR": 391,
        "FN_CLEAR": 6338,
        "IDS": 53,
        "IDP": 0.822,
        "IDR": 0.576,
        "IDF1": 0.677,
        "HOTA": 0.550,  # Does not correspond exactly to original
        "DetA": 0.561,  # Does not correspond exactly to original
        "AssA": 0.540,  # Does not correspond exactly to original
        "LocA": 0.844,  # Does not correspond exactly to original
    }

    print(results)
    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)

    # Check that the results are similar to those obtained with TrackEval
    # Note that they will never be the same, since a different algorithm
    # is used for matching. Nevertheless, it's a good idea to check that
    # we do not get widely different results
    orig_hota_results = {
        "HOTA": 0.5468,
        "DetA": 0.5546,
        "AssA": 0.5411,
        "LocA": 0.8505,
    }

    for key, orig_val in orig_hota_results.items():
        np.testing.assert_array_almost_equal(results[key], orig_val, decimal=2)


test_mot20_01()
