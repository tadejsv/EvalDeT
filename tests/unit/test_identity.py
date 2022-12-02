import numpy as np
import pytest

from evaldet import MOTMetrics, Tracks


def test_empty_frame_hyp(missing_frame_pair):
    m = MOTMetrics()

    gt, hyp = missing_frame_pair
    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )

    assert metrics["id"]["IDFP"] == 0
    assert metrics["id"]["IDFN"] == 1
    assert metrics["id"]["IDTP"] == 1

    assert metrics["id"]["IDP"] == 1
    assert metrics["id"]["IDR"] == 0.5
    assert metrics["id"]["IDF1"] == 2 / 3


def test_missing_frame_gt(missing_frame_pair):
    m = MOTMetrics()

    hyp, gt = missing_frame_pair
    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )

    assert metrics["id"]["IDFP"] == 1
    assert metrics["id"]["IDFN"] == 0
    assert metrics["id"]["IDTP"] == 1

    assert metrics["id"]["IDP"] == 0.5
    assert metrics["id"]["IDR"] == 1
    assert metrics["id"]["IDF1"] == 2 / 3


def test_no_association_made():
    m = MOTMetrics()

    gt = Tracks(ids=[0], frame_nums=[0], detections=np.array([[10, 10, 1, 1]]))
    hyp = Tracks(ids=[0], frame_nums=[0], detections=np.array([[0, 0, 1, 1]]))
    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )

    assert metrics["id"]["IDFP"] == 1
    assert metrics["id"]["IDFN"] == 1
    assert metrics["id"]["IDTP"] == 0

    assert metrics["id"]["IDP"] == 0
    assert metrics["id"]["IDR"] == 0
    assert metrics["id"]["IDF1"] == 0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_dist_threshold(threshold: float):
    m = MOTMetrics(id_dist_threshold=threshold)

    gt = Tracks(
        ids=[0, 1, 2, 3],
        frame_nums=[0] * 4,
        detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
    )

    hyp = Tracks(
        ids=[0, 1, 2, 3],
        frame_nums=[0] * 4,
        detections=np.array(
            [[0, 0, 1, 0.2], [0, 0, 1, 0.4], [0, 0, 1, 0.6], [0, 0, 1, 0.8]]
        ),
    )

    fn_res = {0.3: 3, 0.5: 2, 0.7: 1}

    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )
    assert fn_res[threshold] == metrics["id"]["IDFN"]


def test_association():
    """Test that only one hypotheses gets associated to a ground truth"""
    m = MOTMetrics()

    gt = Tracks(
        frame_nums=[0, 1], ids=[0, 0], detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )
    hyp = Tracks(
        frame_nums=[0, 1], ids=[0, 1], detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )

    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )
    assert metrics["id"]["IDFP"] == 1
    assert metrics["id"]["IDFN"] == 1
    assert metrics["id"]["IDTP"] == 1


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
    m = MOTMetrics()

    gt = Tracks(
        frame_nums=[0, 1], ids=[0, 0], detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )

    hyp_frame_nums = [0, 0, 1, 1] + list(range(2, 10))
    hyp_ids = [0, 1, 0, 1] + [1] * len(range(2, 10))
    hyp_detections = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [10, 10, 1, 1], [0, 0, 1, 1]]
        + [[0, 0, 1, 1]] * len(range(2, 10))
    )
    hyp = Tracks(ids=hyp_ids, frame_nums=hyp_frame_nums, detections=hyp_detections)
    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )
    assert metrics["id"]["IDFP"] == 10
    assert metrics["id"]["IDFN"] == 0
    assert metrics["id"]["IDTP"] == 2


def test_simple_case(simple_case):
    """Test a simple case with 3 frames and 2 detections/gts per frame."""
    m = MOTMetrics()
    gt, hyp = simple_case

    metrics = m.compute(
        gt, hyp, id_metrics=True, clearmot_metrics=True, hota_metrics=True
    )

    assert metrics["id"]["IDFP"] == 2
    assert metrics["id"]["IDFN"] == 2
    assert metrics["id"]["IDTP"] == 4

    assert metrics["id"]["IDP"] == 2 / 3
    assert metrics["id"]["IDR"] == 2 / 3
    assert metrics["id"]["IDF1"] == 2 / 3
