import numpy as np

from evaldet import Tracks
from evaldet.mot_metrics.hota import calculate_hota_metrics


def test_hyp_missing_frame():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_hota_metrics(gt, hyp)

    assert metrics["DetA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["HOTA"] == 0.5
    assert metrics["LocA"] == 1.0
    for m in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics[m].var() == 0


def test_gt_missing_frame():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_hota_metrics(gt, hyp)

    assert metrics["DetA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["HOTA"] == 0.5
    assert metrics["LocA"] == 1.0
    for m in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics[m].var() == 0


def test_no_matches():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[1, 1, 2, 2]]))
    metrics = calculate_hota_metrics(gt, hyp)

    assert metrics["DetA"] == 0
    assert metrics["AssA"] == 0
    assert metrics["HOTA"] == 0
    assert metrics["LocA"] == 1.0
    for m in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics[m].var() == 0


def test_alphas():
    gt = Tracks()
    gt.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))

    hyp = Tracks()
    hyp.add_frame(0, [0, 1], np.array([[0.1, 0.1, 1.1, 1.1], [1.2, 1.2, 2.2, 2.2]]))
    metrics = calculate_hota_metrics(gt, hyp)

    np.testing.assert_almost_equal(metrics["AssA"], 0.6842105263157, 5)
    np.testing.assert_almost_equal(metrics["DetA"], 0.5438596491228, 5)
    np.testing.assert_almost_equal(metrics["HOTA"], 0.5952316356188, 5)
    np.testing.assert_almost_equal(metrics["LocA"], 0.7317558602388, 5)

    for a_ind, alpha in enumerate(metrics["alphas_HOTA"]):
        if alpha < 0.31932773:
            assert metrics["HOTA_alpha"][a_ind] == 0
            assert metrics["AssA_alpha"][a_ind] == 0
            assert metrics["DetA_alpha"][a_ind] == 0
            assert metrics["LocA_alpha"][a_ind] == 1.0
        elif 0.52941176 > alpha >= 0.31932773:
            assert metrics["HOTA_alpha"][a_ind] == np.sqrt(1 / 3)
            assert metrics["AssA_alpha"][a_ind] == 1.0
            assert metrics["DetA_alpha"][a_ind] == 1 / 3
            np.testing.assert_almost_equal(
                metrics["LocA_alpha"][a_ind], 1 - 0.31932773, 5
            )
        else:
            assert metrics["HOTA_alpha"][a_ind] == 1
            assert metrics["AssA_alpha"][a_ind] == 1.0
            assert metrics["DetA_alpha"][a_ind] == 1
            np.testing.assert_almost_equal(
                metrics["LocA_alpha"][a_ind], 1 - (0.31932773 + 0.52941176) / 2, 5
            )


def test_priority_matching_1():
    pass


def test_priority_matching_2():
    pass


def test_example_1a():
    """Example A from figure 1 in the paper"""
    gt = Tracks()
    for i in range(20):
        gt.add_frame(i, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    for i in range(10):
        hyp.add_frame(i, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_hota_metrics(gt, hyp)

    assert metrics["HOTA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["DetA"] == 0.5
    assert metrics["LocA"] == 1.0


def test_example_1b():
    """Example B from figure 1 in the paper"""
    gt = Tracks()
    for i in range(20):
        gt.add_frame(i, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    for i in range(7):
        hyp.add_frame(i, [0], np.array([[0, 0, 1, 1]]))

    for i in range(7, 14):
        hyp.add_frame(i, [1], np.array([[0, 0, 1, 1]]))
    metrics = calculate_hota_metrics(gt, hyp)

    np.testing.assert_array_almost_equal(metrics["HOTA"], 0.5, 2)
    np.testing.assert_array_almost_equal(metrics["AssA"], 0.35, 2)
    np.testing.assert_array_almost_equal(metrics["DetA"], 0.70, 2)
    assert metrics["LocA"] == 1.0


def test_example_1c():
    """Example C from figure 1 in the paper"""
    gt = Tracks()
    for i in range(20):
        gt.add_frame(i, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    for i in range(5):
        hyp.add_frame(i, [0], np.array([[0, 0, 1, 1]]))

    for i in range(5, 10):
        hyp.add_frame(i, [1], np.array([[0, 0, 1, 1]]))

    for i in range(10, 15):
        hyp.add_frame(i, [2], np.array([[0, 0, 1, 1]]))

    for i in range(15, 20):
        hyp.add_frame(i, [3], np.array([[0, 0, 1, 1]]))
    metrics = calculate_hota_metrics(gt, hyp)

    np.testing.assert_array_almost_equal(metrics["HOTA"], 0.5, 2)
    np.testing.assert_array_almost_equal(metrics["AssA"], 0.25, 2)
    np.testing.assert_array_almost_equal(metrics["DetA"], 1, 2)
    assert metrics["LocA"] == 1.0


def test_simple_case():
    """Test a simple case with 3 frames and 2 detections/gts per frame."""
    gt = Tracks()
    gt.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))
    gt.add_frame(1, [0, 1], np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))
    gt.add_frame(2, [0, 1], np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))

    hyp = Tracks()
    hyp.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))
    hyp.add_frame(1, [0, 1], np.array([[0.1, 0.1, 1.1, 1.1], [1, 1, 2, 2]]))
    hyp.add_frame(2, [2, 1], np.array([[0.05, 0.05, 1.05, 1.05], [2, 2, 3, 3]]))

    # metrics = calculate_hota_metrics(gt, hyp)
