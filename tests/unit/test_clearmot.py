import numpy as np
import pytest

from evaldet import MOTMetrics, Tracks


def test_missing_frame_hyp():
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )

    assert metrics["clearmot"]["FN_CLEAR"] == 1
    assert metrics["clearmot"]["FP_CLEAR"] == 0
    assert metrics["clearmot"]["IDSW"] == 0


def test_missing_frame_gt():
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )

    assert metrics["clearmot"]["IDSW"] == 0
    assert metrics["clearmot"]["FN_CLEAR"] == 0
    assert metrics["clearmot"]["FP_CLEAR"] == 1


def test_no_association_made():
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0], np.array([[10, 10, 11, 11]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )

    assert metrics["clearmot"]["IDSW"] == 0
    assert metrics["clearmot"]["FN_CLEAR"] == 1
    assert metrics["clearmot"]["FP_CLEAR"] == 1
    assert metrics["clearmot"]["MOTA"] == -1  # Stange but ok
    assert np.isnan(metrics["clearmot"]["MOTP"])


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_dist_threshold(threshold: float):
    m = MOTMetrics(clearmot_dist_threshold=threshold)
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

    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )
    assert fn_res[threshold] == metrics["clearmot"]["FN_CLEAR"]


def test_sticky_association():
    """Test that as long as distance is below threshold, the association does
    not switch, even if a detection with better IoU score appears.
    """
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [0, 1], np.array([[0.1, 0.1, 1.1, 1.1], [0, 0, 1, 1]]))

    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )
    assert metrics["clearmot"]["FN_CLEAR"] == 0
    assert metrics["clearmot"]["IDSW"] == 0
    assert metrics["clearmot"]["FP_CLEAR"] == 1


def test_mismatch():
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(1, [1], np.array([[0, 0, 1, 1]]))

    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )
    assert metrics["clearmot"]["FN_CLEAR"] == 0
    assert metrics["clearmot"]["IDSW"] == 1
    assert metrics["clearmot"]["FP_CLEAR"] == 0


def test_persistent_mismatch():
    """Test that association (and therefore mismatch) persists even
    when the first matched hypothesis is gone, as long as another one
    is not assigned."""
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(1, [0], np.array([[0, 0, 1, 1]]))
    gt.add_frame(2, [0], np.array([[0, 0, 1, 1]]))

    hyp = Tracks()
    hyp.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    hyp.add_frame(2, [1], np.array([[0, 0, 1, 1]]))

    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )
    assert metrics["clearmot"]["FN_CLEAR"] == 1
    assert metrics["clearmot"]["IDSW"] == 1
    assert metrics["clearmot"]["FP_CLEAR"] == 0


def test_simple_case():
    """Test a simple case with 3 frames and 2 detections/gts per frame."""
    m = MOTMetrics()

    gt = Tracks()
    gt.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))
    gt.add_frame(1, [0, 1], np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))
    gt.add_frame(2, [0, 1], np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))

    hyp = Tracks()
    hyp.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))
    hyp.add_frame(1, [0, 1], np.array([[0.1, 0.1, 1.1, 1.1], [1, 1, 2, 2]]))
    hyp.add_frame(2, [2, 1], np.array([[0.05, 0.05, 1.05, 1.05], [2, 2, 3, 3]]))

    metrics = m.compute(
        gt, hyp, clearmot_metrics=True, id_metrics=False, hota_metrics=False
    )
    assert metrics["clearmot"]["FN_CLEAR"] == 1
    assert metrics["clearmot"]["IDSW"] == 1
    assert metrics["clearmot"]["FP_CLEAR"] == 1
    assert metrics["clearmot"]["MOTA"] == 0.5
    assert metrics["clearmot"]["MOTP"] == 0.0994008537355717
