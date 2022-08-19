import numpy as np

from evaldet import Tracks, compute_mot_metrics
from evaldet.utils import preprocess_mot_1720


def test_tud_campus():
    gt = Tracks.from_mot("tests/data/integration/tud_campus_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/tud_campus_hyp.csv")

    results = compute_mot_metrics(
        clearmot_metrics=True,
        hota_metrics=True,
        id_metrics=True,
        ground_truth=gt,
        detections=hyp,
    )
    exp_results = {
        "MOTA": 0.526,
        "MOTP": 0.277,
        "FP_CLEAR": 13,
        "FN_CLEAR": 150,
        "IDSW": 7,
        "IDP": 0.730,
        "IDR": 0.451,
        "IDF1": 0.558,
        "HOTA": 0.393,  # Does not correspond exactly to original
        "DetA": 0.425,  # Does not correspond exactly to original
        "AssA": 0.365,  # Does not correspond exactly to original
        "LocA": 0.768,  # Does not correspond exactly to original
    }

    for key in exp_results:
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

    results = compute_mot_metrics(
        clearmot_metrics=True,
        hota_metrics=True,
        id_metrics=True,
        ground_truth=gt,
        detections=hyp,
    )
    exp_results = {
        "MOTA": 0.564,
        "MOTP": 0.346,
        "FP_CLEAR": 45,
        "FN_CLEAR": 452,
        "IDSW": 7,
        "IDP": 0.820,
        "IDR": 0.531,
        "IDF1": 0.645,
        "HOTA": 0.399,  # Does not correspond exactly to original
        "DetA": 0.399,  # Does not correspond exactly to original
        "AssA": 0.404,  # Does not correspond exactly to original
        "LocA": 0.733,  # Does not correspond exactly to original
    }

    for key in exp_results:
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


def test_mot20_01():
    gt = Tracks.from_mot_gt("tests/data/integration/MOT20-01_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/MOT20-01_MPNTrack_hyp.csv")
    preprocess_mot_1720(gt, hyp)

    results = compute_mot_metrics(
        clearmot_metrics=True,
        hota_metrics=True,
        id_metrics=True,
        ground_truth=gt,
        detections=hyp,
    )
    exp_results = {
        "MOTA": 0.659,
        "MOTP": 1 - 0.833,
        "FP_CLEAR": 391,
        "FN_CLEAR": 6338,
        "IDSW": 53,
        "IDP": 0.822,
        "IDR": 0.576,
        "IDF1": 0.677,
        "HOTA": 0.550,  # Does not correspond exactly to original
        "DetA": 0.561,  # Does not correspond exactly to original
        "AssA": 0.540,  # Does not correspond exactly to original
        "LocA": 0.844,  # Does not correspond exactly to original
    }

    for key in exp_results:
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


def test_mot20_03():
    gt = Tracks.from_mot_gt("tests/data/integration/MOT20-03_gt.csv")
    hyp = Tracks.from_mot("tests/data/integration/MOT20-03_MPNTrack_hyp.csv")
    preprocess_mot_1720(gt, hyp)

    results = compute_mot_metrics(
        clearmot_metrics=True,
        hota_metrics=True,
        id_metrics=True,
        ground_truth=gt,
        detections=hyp,
    )
    exp_results = {
        "MOTA": 0.78031,
        "MOTP": 1 - 0.81614,
        "FP_CLEAR": 3083,  # Original is 3977, unexplained diff
        "FN_CLEAR": 65542,  # Original is 65536, unexplained diff
        "IDSW": 293,  # Original is 294, unexplained diff
        "IDP": 0.88783,
        "IDR": 0.71104,
        "IDF1": 0.78966,
        "HOTA": 0.6206,  # Does not correspond exactly to original
        "DetA": 0.6365,  # Does not correspond exactly to original
        "AssA": 0.6056,  # Does not correspond exactly to original
        "LocA": 0.8318,  # Does not correspond exactly to original
    }

    for key in exp_results:
        np.testing.assert_array_almost_equal(results[key], exp_results[key], decimal=3)

    # Check that the results are similar to those obtained with TrackEval
    # Note that they will never be the same, since a different algorithm
    # is used for matching. Nevertheless, it's a good idea to check that
    # we do not get widely different results
    orig_hota_results = {
        "HOTA": 0.620,
        "DetA": 0.6293,
        "AssA": 0.6074,
        "LocA": 0.8360,
    }

    for key, orig_val in orig_hota_results.items():
        np.testing.assert_array_almost_equal(results[key], orig_val, decimal=2)
