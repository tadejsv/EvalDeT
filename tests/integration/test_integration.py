import numpy as np

from evaldet import Tracks, compute_mot_metrics

_TUD_METRICS = ("MOTA", "MOTP", "FP_CLEAR", "FN_CLEAR", "IDS", "IDP", "IDR", "IDF1")


def test_tud_campus():
    gt = Tracks.from_mot("tests/data/integration/tud_campus_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/tud_campus_hyp.csv")

    results = compute_mot_metrics(metrics=_TUD_METRICS, ground_truth=gt, detections=hyp)
    exp_results = {
        "MOTA": 0.526,
        "MOTP": 0.277,
        "FP_CLEAR": 13,
        "FN_CLEAR": 150,
        "IDS": 7,
        "IDP": 0.730,
        "IDR": 0.451,
        "IDF1": 0.558,
    }

    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)


def test_tud_stadtmitte():
    gt = Tracks.from_mot("tests/data/integration/tud_stadtmitte_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/tud_stadtmitte_hyp.csv")

    results = compute_mot_metrics(metrics=_TUD_METRICS, ground_truth=gt, detections=hyp)
    exp_results = {
        "MOTA": 0.564,
        "MOTP": 0.346,
        "FP_CLEAR": 45,
        "FN_CLEAR": 452,
        "IDS": 7,
        "IDP": 0.820,
        "IDR": 0.531,
        "IDF1": 0.645,
    }

    print(results)
    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)
