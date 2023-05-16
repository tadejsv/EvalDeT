import numpy as np
import numpy.testing as npt

from evaldet.det.coco import evaluate_image
from evaldet.dist import iou_dist


def test_coco_evaluate_image_both_empty():
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 0), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        (0, float("inf")),
        0.5,
    )

    assert matched.size == 0
    assert ignored_pred.size == 0
    assert ignored_gt.size == 0
    assert n_gt == 0


def test_coco_evaluate_image_preds_empty():
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        np.zeros((0, 4), dtype=np.float32),
        np.random.rand(1, 4).astype(np.float32),
        np.zeros((0, 1), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        (0, float("inf")),
        0.5,
    )

    assert matched.size == 0
    assert ignored_pred.size == 0
    npt.assert_array_equal(ignored_gt, np.array([False], dtype=np.bool_))
    assert n_gt == 1


def test_coco_evaluate_image_gts_empty():
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        np.random.rand(1, 4).astype(np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((1, 0), dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        (0, float("inf")),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([-1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([False], dtype=np.bool_))
    assert ignored_gt.size == 0
    assert n_gt == 0


def test_coco_evaluate_image_gts_all_ignored():
    preds_bbox = 10 + np.random.rand(1, 4).astype(np.float32)
    gts_bbox = np.random.rand(2, 4).astype(np.float32)  # Size will be < 1 * 1
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.array([0.5], dtype=np.float32),
        (1, float("inf")),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([-1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([False], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([True, True], dtype=np.bool_))
    assert n_gt == 0


def test_coco_evaluate_image_gts_small_ignored():
    preds_bbox = 10 + np.random.rand(1, 4).astype(np.float32)
    gts_bbox = np.array([[1, 1, 0.1, 1], [2, 2, 10, 10]], dtype=np.float32)
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.random.rand(1).astype(np.float32),
        (1, float("inf")),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([-1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([False], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([True, False], dtype=np.bool_))
    assert n_gt == 1


def test_coco_evaluate_image_gts_large_ignored():
    preds_bbox = np.random.rand(1, 4).astype(np.float32)
    gts_bbox = np.array([[1, 1, 0.1, 1], [2, 2, 10, 10]], dtype=np.float32)
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.random.rand(1).astype(np.float32),
        (0, 1),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([-1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([False], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([False, True], dtype=np.bool_))
    assert n_gt == 1


def test_coco_evaluate_image_preds_ignored_matched():
    preds_bbox = np.array([[2, 2, 10, 10]], dtype=np.float32)
    gts_bbox = np.array([[1, 1, 0.1, 1], [2, 2, 10, 10]], dtype=np.float32)
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.random.rand(1).astype(np.float32),
        (0, 1),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([True], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([False, True], dtype=np.bool_))
    assert n_gt == 1


def test_coco_evaluate_image_preds_unmatched_ignored():
    preds_bbox = np.array([[2, 2, 10, 10]], dtype=np.float32)
    gts_bbox = np.array([[1, 1, 0.1, 1]], dtype=np.float32)
    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.random.rand(1).astype(np.float32),
        (0, 1),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([-1], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([True], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([False], dtype=np.bool_))
    assert n_gt == 1


def test_coco_evaluate_image_matching_1():
    """Simple matching test with unordered preds and one ignored gt"""
    preds_bbox = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]], dtype=np.float32)
    gts_bbox = np.array([[1, 1, 1, 1], [3, 3, 10, 10], [0, 0, 1, 1]], dtype=np.float32)

    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.array([0.1, 0.9, 0.5], dtype=np.float32),
        (0, 2),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([2, 0, -1], dtype=np.int32))
    npt.assert_array_equal(
        ignored_pred, np.array([False, False, False], dtype=np.bool_)
    )
    npt.assert_array_equal(ignored_gt, np.array([False, True, False], dtype=np.bool_))
    assert n_gt == 2


def test_coco_evaluate_image_matching_2():
    """
    Simple matching test with unordered preds and unmatched gt/pred pair - due to low
    IoU.
    """
    preds_bbox = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]], dtype=np.float32)
    gts_bbox = np.array(
        [[1, 1, 1, 1], [0, 0, 1, 1], [2.5, 2.5, 1, 1]], dtype=np.float32
    )

    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.array([0.1, 0.9, 0.5], dtype=np.float32),
        (0, 2),
        0.5,
    )

    npt.assert_array_equal(matched, np.array([1, 0, -1], dtype=np.int32))
    npt.assert_array_equal(
        ignored_pred, np.array([False, False, False], dtype=np.bool_)
    )
    npt.assert_array_equal(ignored_gt, np.array([False, False, False], dtype=np.bool_))
    assert n_gt == 3


def test_coco_evaluate_image_matching_3():
    """
    Simple matching test where high confidence prediction takes precedence over a
    high IoU prediction
    """
    preds_bbox = np.array([[1, 1, 1, 1], [1.2, 1.2, 1, 1]], dtype=np.float32)
    gts_bbox = np.array([[1, 1, 1, 1]], dtype=np.float32)

    matched, ignored_pred, ignored_gt, n_gt = evaluate_image(
        preds_bbox,
        gts_bbox,
        1 - iou_dist(preds_bbox, gts_bbox),
        np.array([0.1, 0.9], dtype=np.float32),
        (0, 2),
        0.1,
    )

    npt.assert_array_equal(matched, np.array([-1, 0], dtype=np.int32))
    npt.assert_array_equal(ignored_pred, np.array([False, False], dtype=np.bool_))
    npt.assert_array_equal(ignored_gt, np.array([False], dtype=np.bool_))
    assert n_gt == 1


def test_coco_pr_curve_no_gts():
    pass


def test_coco_pr_curve_gts_all_ignored():
    pass


def test_coco_pr_curve_no_preds():
    pass


def test_coco_pr_curve_preds_all_ignored():
    pass


def test_coco_pr_curve_frame_no_preds():
    pass


def test_coco_pr_curve_frame_no_gts():
    pass


def test_coco_pr_curve_example_1():
    """Simple 2 frame example, one unmatched GT and one unmatched pred"""
    pass


def test_coco_pr_curve_example_2():
    """Simple 2 frame example, one ignored GT and ignored unmatched pred"""
    pass


def test_coco_pr_curve_example_3():
    """Simple 3 frame example with some more preds and GTs"""
    pass
