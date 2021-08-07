import numpy as np
import pytest

from evaldet import Tracks
from evaldet.mot_metrics.identity import calculate_id_metrics


def test_empty_frame_hyp():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_id_metrics(gt, hyp)

    assert metrics["IDFP"] == 0
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 1

    assert metrics["IDP"] == 1
    assert metrics["IDR"] == 0.5
    assert metrics["IDF1"] == 2 / 3


def test_missing_frame_gt():
    gt = Tracks()
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_id_metrics(gt, hyp)

    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 0
    assert metrics["IDTP"] == 1

    assert metrics["IDP"] == 0.5
    assert metrics["IDR"] == 1
    assert metrics["IDF1"] == 2 / 3


def test_no_association_made():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[10, 10, 11, 11]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_id_metrics(gt, hyp)

    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 0

    assert metrics["IDP"] == 0
    assert metrics["IDR"] == 0
    assert metrics["IDF1"] == 0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_dist_threshold(threshold: float):
    gt = Tracks()
    gt.add_frame(
        0,
        [0, 1, 2, 3],
        np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
    )

    hyp = Tracks()
    hyp.add_frame(
        0,
        [0, 1, 2, 3],
        np.array([[0, 0, 1, 0.2], [0, 0, 1, 0.4], [0, 0, 1, 0.6], [0, 0, 1, 0.8]]),
    )

    fn_res = {0.3: 3, 0.5: 2, 0.7: 1}

    metrics = calculate_id_metrics(gt, hyp, dist_threshold=threshold)
    assert fn_res[threshold] == metrics["IDFN"]


def test_association():
    """Test that only one hypotheses gets associated to a ground truth"""
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [1], np.array([[0, 0, 1, 1]]))

    metrics = calculate_id_metrics(gt, hyp)
    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 1


def test_proper_cost_function():
    """This test makes sure than when making associations, costs of
    hypotheses that are not matched are also taken into account.

    The situation is like this: we have 1 gt and 2 hyps. The gt is
    present in 2 frames, and the 1st hyp matches it in 1 frame, but
    misses in another. The 2nd hyp mathches it in both frames, but
    is also present in 8 other frames where there is no gt - so
    these are all FP frames. If we make the cost matrix [n_gt, n_hyp],
    this will lead to erroneous matching of gt with hypothesis 1, while
    it should have matched hypothesis 2
    """
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [0, 0, 1, 1]]))
    hyp.add_frame(1, [0, 1], np.array([[10, 10, 11, 11], [0, 0, 1, 1]]))
    for i in range(2, 10):
        hyp.add_frame(i, [1], np.array([[0, 0, 1, 1]]))

    metrics = calculate_id_metrics(gt, hyp)
    assert metrics["IDFP"] == 10
    assert metrics["IDFN"] == 0
    assert metrics["IDTP"] == 2


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

    metrics = calculate_id_metrics(gt, hyp)

    assert metrics["IDFP"] == 2
    assert metrics["IDFN"] == 2
    assert metrics["IDTP"] == 4

    assert metrics["IDP"] == 2 / 3
    assert metrics["IDR"] == 2 / 3
    assert metrics["IDF1"] == 2 / 3
