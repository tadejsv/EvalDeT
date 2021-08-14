import numpy as np

from evaldet import Tracks, compute_mot_metrics

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
        "HOTA": 0.393,  # Does not correspond exactly to iriginal
        "DetA": 0.425,  # Does not correspond exactly to iriginal
        "AssA": 0.365,  # Does not correspond exactly to iriginal
        "LocA": 0.768,  # Does not correspond exactly to iriginal
    }

    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)

    # Check that the results are similar to those obtained with TrackEval
    # Note that they will never be the same, since a different algorithm
    # is used for matching. Nevertheless, it's a good idea to check that
    # we do not get widely different results
    orig_hota_results = {
        "HOTA": 0.3914,
        "DetA": 0.4181,
        "AssA": 0.3691,
        "LocA": 0.7701,
    }

    for key, orig_val in orig_hota_results.items():
        np.testing.assert_array_almost_equal(results[key], orig_val, decimal=2)


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
        "HOTA": 0.399,  # Does not correspond exactly to iriginal
        "DetA": 0.399,  # Does not correspond exactly to iriginal
        "AssA": 0.404,  # Does not correspond exactly to iriginal
        "LocA": 0.733,  # Does not correspond exactly to iriginal
    }

    for key in results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)

    # Check that the results are similar to those obtained with TrackEval
    # Note that they will never be the same, since a different algorithm
    # is used for matching. Nevertheless, it's a good idea to check that
    # we do not get widely different results
    orig_hota_results = {
        "HOTA": 0.3979,
        "DetA": 0.3923,
        "AssA": 0.4085,
        "LocA": 0.7375,
    }

    for key, orig_val in orig_hota_results.items():
        np.testing.assert_array_almost_equal(results[key], orig_val, decimal=2)
