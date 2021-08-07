import numpy as np

from evaldet import Tracks, compute_mot_metrics

_TUD_METRICS = ("MOTA", "MOTP", "FP", "FN", "IDS")


def test_tud_campus():
    gt = Tracks.from_mot("tests/data/integration/tud_campus_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/tud_campus_hyp.csv")

    results = compute_mot_metrics(metrics=_TUD_METRICS, ground_truth=gt, detections=hyp)
    exp_results = {"MOTA": 0.526, "MOTP": 0.277, "FP": 13, "FN": 150, "IDS": 7}

    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)


def test_tud_stadtmitte():
    gt = Tracks.from_mot("tests/data/integration/tud_stadtmitte_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/tud_stadtmitte_hyp.csv")

    results = compute_mot_metrics(metrics=_TUD_METRICS, ground_truth=gt, detections=hyp)
    exp_results = {"MOTA": 0.564, "MOTP": 0.346, "FP": 45, "FN": 452, "IDS": 7}

    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)
