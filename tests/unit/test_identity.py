import numpy as np
import pytest

from evaldet.mot.identity import calculate_id_metrics
from evaldet.tracks import Tracks
from evaldet.mot.motmetrics import _compute_ious


def test_empty_frame_hyp(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    gt, hyp = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 0
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 1

    assert metrics["IDP"] == 1
    assert metrics["IDR"] == 0.5
    assert metrics["IDF1"] == 2 / 3


def test_missing_frame_gt(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    hyp, gt = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 0
    assert metrics["IDTP"] == 1

    assert metrics["IDP"] == 0.5
    assert metrics["IDR"] == 1
    assert metrics["IDF1"] == 2 / 3


def test_no_association_made() -> None:
    gt = Tracks(ids=[0], frame_nums=[0], detections=np.array([[10, 10, 1, 1]]))
    hyp = Tracks(ids=[0], frame_nums=[0], detections=np.array([[0, 0, 1, 1]]))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 0

    assert metrics["IDP"] == 0
    assert metrics["IDR"] == 0
    assert metrics["IDF1"] == 0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_dist_threshold(threshold: float) -> None:
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

    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious, dist_threshold=threshold)

    assert fn_res[threshold] == metrics["IDFN"]


def test_association() -> None:
    """Test that only one hypotheses gets associated to a ground truth"""
    gt = Tracks(
        frame_nums=[0, 1], ids=[0, 0], detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )
    hyp = Tracks(
        frame_nums=[0, 1], ids=[0, 1], detections=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )

    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 1
    assert metrics["IDFN"] == 1
    assert metrics["IDTP"] == 1


def test_proper_cost_function() -> None:
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
    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 10
    assert metrics["IDFN"] == 0
    assert metrics["IDTP"] == 2


def test_simple_case(simple_case: tuple[Tracks, Tracks]) -> None:
    """Test a simple case with 3 frames and 2 detections/gts per frame."""
    gt, hyp = simple_case
    ious = _compute_ious(gt, hyp)
    metrics = calculate_id_metrics(gt, hyp, ious=ious)

    assert metrics["IDFP"] == 2
    assert metrics["IDFN"] == 2
    assert metrics["IDTP"] == 4

    assert metrics["IDP"] == 2 / 3
    assert metrics["IDR"] == 2 / 3
    assert metrics["IDF1"] == 2 / 3
