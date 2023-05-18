import numpy as np
import pytest

from evaldet.mot.clearmot import calculate_clearmot_metrics
from evaldet.mot.motmetrics import _compute_ious
from evaldet.tracks import Tracks


def test_missing_frame_hyp(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    gt, hyp = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["FN_CLEAR"] == 1
    assert metrics["FP_CLEAR"] == 0
    assert metrics["IDSW"] == 0


def test_missing_frame_gt(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    hyp, gt = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["IDSW"] == 0
    assert metrics["FN_CLEAR"] == 0
    assert metrics["FP_CLEAR"] == 1


def test_no_association_made() -> None:
    gt = Tracks(ids=[0], frame_nums=[0], detections=np.array([[10, 10, 11, 11]]))
    hyp = Tracks(ids=[0], frame_nums=[0], detections=np.array([[0, 0, 1, 1]]))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["IDSW"] == 0
    assert metrics["FN_CLEAR"] == 1
    assert metrics["FP_CLEAR"] == 1
    assert metrics["MOTA"] == -1  # Stange but ok
    assert np.isnan(metrics["MOTP"])


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
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious, dist_threshold=threshold)

    assert metrics is not None
    assert fn_res[threshold] == metrics["FN_CLEAR"]


def test_sticky_association() -> None:
    """Test that as long as distance is below threshold, the association does
    not switch, even if a detection with better IoU score appears.
    """
    gt = Tracks(ids=[0, 0], frame_nums=[0, 1], detections=np.array([[0, 0, 1, 1]] * 2))
    hyp = Tracks(
        ids=[0, 0, 1],
        frame_nums=[0, 1, 1],
        detections=np.array([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1], [0, 0, 1, 1]]),
    )
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["FN_CLEAR"] == 0
    assert metrics["IDSW"] == 0
    assert metrics["FP_CLEAR"] == 1


def test_mismatch() -> None:
    gt = Tracks(ids=[0, 0], frame_nums=[0, 1], detections=np.array([[0, 0, 1, 1]] * 2))
    hyp = Tracks(ids=[0, 1], frame_nums=[0, 1], detections=np.array([[0, 0, 1, 1]] * 2))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["FN_CLEAR"] == 0
    assert metrics["IDSW"] == 1
    assert metrics["FP_CLEAR"] == 0


def test_persistent_mismatch() -> None:
    """Test that association (and therefore mismatch) persists even
    when the first matched hypothesis is gone, as long as another one
    is not assigned."""
    gt = Tracks(
        ids=[0] * 3, frame_nums=[0, 1, 2], detections=np.array([[0, 0, 1, 1]] * 3)
    )
    hyp = Tracks(ids=[0, 1], frame_nums=[0, 2], detections=np.array([[0, 0, 1, 1]] * 2))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["FN_CLEAR"] == 1
    assert metrics["IDSW"] == 1
    assert metrics["FP_CLEAR"] == 0


def test_simple_case(simple_case: tuple[Tracks, Tracks]) -> None:
    """Test a simple case with 3 frames and 2 detections/gts per frame."""
    gt, hyp = simple_case
    ious = _compute_ious(gt, hyp)
    metrics = calculate_clearmot_metrics(gt, hyp, ious=ious)

    assert metrics["FN_CLEAR"] == 1
    assert metrics["IDSW"] == 1
    assert metrics["FP_CLEAR"] == 1
    assert metrics["MOTA"] == 0.5
    assert metrics["MOTP"] == pytest.approx(0.0994008537355717)
