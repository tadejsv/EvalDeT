import numpy as np
import pytest

from evaldet import Tracks
from evaldet.mot_metrics.clearmot import calculate_clearmot_metrics


def test_empty_hypotheses():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_clearmot_metrics(gt, hyp)

    assert metrics["FN_CLEAR"] == 1
    assert metrics["FP_CLEAR"] == 0
    assert metrics["IDS"] == 0


def test_missing_frame_gt():
    gt = Tracks()
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_clearmot_metrics(gt, hyp)

    assert metrics["IDS"] == 0
    assert metrics["FN_CLEAR"] == 0
    assert metrics["FP_CLEAR"] == 1


def test_no_association_made():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[10, 10, 11, 11]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_clearmot_metrics(gt, hyp)

    assert metrics["IDS"] == 0
    assert metrics["FN_CLEAR"] == 1
    assert metrics["FP_CLEAR"] == 1
    assert metrics["MOTA"] == -1  # Stange but ok
    assert np.isnan(metrics["MOTP"])


def test_missing_frame_hypotheses():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    metrics = calculate_clearmot_metrics(gt, hyp)

    assert metrics["IDS"] == 0
    assert metrics["FN_CLEAR"] == 1
    assert metrics["FP_CLEAR"] == 0


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

    metrics = calculate_clearmot_metrics(gt, hyp, dist_threshold=threshold)
    assert fn_res[threshold] == metrics["FN_CLEAR"]


def test_sticky_association():
    """Test that as long as distance is below threshold, the association does
    not switch, even if a detection with better IoU score appears.
    """
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0, 1], np.array([[0.1, 0.1, 1.1, 1.1], [0, 0, 1, 1]]))

    metrics = calculate_clearmot_metrics(gt, hyp)
    assert metrics["FN_CLEAR"] == 0
    assert metrics["IDS"] == 0
    assert metrics["FP_CLEAR"] == 1


def test_mismatch():
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [1], np.array([[0, 0, 1, 1]]))

    metrics = calculate_clearmot_metrics(gt, hyp)
    assert metrics["FN_CLEAR"] == 0
    assert metrics["IDS"] == 1
    assert metrics["FP_CLEAR"] == 0


def test_persistent_mismatch():
    """Test that association (and therefore mismatch) persists even
    when the first matched hypothesis is gone, as long as another one
    is not assigned."""
    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(2, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(2, [1], np.array([[0, 0, 1, 1]]))

    metrics = calculate_clearmot_metrics(gt, hyp)
    assert metrics["FN_CLEAR"] == 1
    assert metrics["IDS"] == 1
    assert metrics["FP_CLEAR"] == 0


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

    metrics = calculate_clearmot_metrics(gt, hyp)
    assert metrics["FN_CLEAR"] == 1
    assert metrics["IDS"] == 1
    assert metrics["FP_CLEAR"] == 1
    assert metrics["MOTA"] == 0.5
    assert metrics["MOTP"] == 0.0994008537355717
