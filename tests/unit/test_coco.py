import numpy as np
import numpy.testing as npt
import pytest

from evaldet.det.coco import calculate_pr_curve, evaluate_image
from evaldet.dist import iou_dist


def _ious_to_numba(ious: dict[int, np.ndarray]):
    pass


def _frame_dict_to_numba(fdict: dict[int, tuple[int, int]]):
    pass


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
    prec_rec = calculate_pr_curve(
        preds_bbox=np.random.rand(1, 4).astype(np.float32),
        gts_bbox=np.zeros((0, 4), dtype=np.float32),
        ious={0: np.zeros((1, 0), dtype=np.float32)},
        preds_conf=np.array([0.5], dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1)},
        frame_ind_dict_gts={},
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.array([[0], [np.nan]], dtype=np.float32))


def test_coco_pr_curve_gts_all_ignored():
    prec_rec = calculate_pr_curve(
        preds_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 0.1, 1]], dtype=np.float32),
        ious={0: np.zeros((1, 0), dtype=np.float32)},
        preds_conf=np.array([0.5], dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1)},
        frame_ind_dict_gts={},
        area_range=(0.5, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.array([[0], [np.nan]], dtype=np.float32))


def test_coco_pr_curve_no_preds():
    prec_rec = calculate_pr_curve(
        preds_bbox=np.zeros((0, 4), dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious={0: np.zeros((0, 1), dtype=np.float32)},
        preds_conf=np.zeros((0,), dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1)},
        frame_ind_dict_gts={},
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.zeros((2, 0), dtype=np.float32))


def test_coco_pr_curve_preds_all_ignored():
    prec_rec = calculate_pr_curve(
        preds_bbox=np.array([[0, 0, 0.1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious={0: np.zeros((1, 0), dtype=np.float32)},
        preds_conf=np.array([0.5], dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1)},
        frame_ind_dict_gts={},
        area_range=(0.5, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.zeros((2, 0), dtype=np.float32))


def test_coco_pr_curve_frame_no_preds():
    """Preds missing on one frame"""
    prec_rec = calculate_pr_curve(
        preds_bbox=np.array(
            [
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        gts_bbox=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        ious={
            0: np.array([[1]], dtype=np.float32),
            1: np.zeros((0, 1), dtype=np.float32),
        },
        preds_conf=np.array([0.5], dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1)},
        frame_ind_dict_gts={0: (0, 1), 1: (1, 2)},
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.array([[1], [0.5]], dtype=np.float32))


def test_coco_pr_curve_frame_no_gts():
    prec_rec = calculate_pr_curve(
        preds_bbox=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious={
            0: np.array([[1]], dtype=np.float32),
            1: np.zeros((1, 0), dtype=np.float32),
        },
        preds_conf=np.array([0.5, 0.6], dtype=np.float32),
        frame_ind_dict_preds={0: (0, 1), 1: (1, 2)},
        frame_ind_dict_gts={0: (0, 1)},
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(prec_rec, np.array([[0, 0.5], [0, 1]], dtype=np.float32))


def test_coco_pr_curve_example_1():
    """Simple 2 frame example, one unmatched GT and one unmatched pred"""
    preds_bbox = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
        ],
        dtype=np.float32,
    )
    gts_bbox = np.array(
        [
            [0.1, 0.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [2.1, 2.1, 1, 1],
            [0.1, 0.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [2.5, 2.5, 1, 1],
        ],
        dtype=np.float32,
    )
    preds_conf = np.array([0.7, 0.9, 0.8, 0.81, 0.6, 0.85], dtype=np.float32)
    ious = {
        0: 1 - iou_dist(preds_bbox[:3], gts_bbox[:3]),
        1: 1 - iou_dist(preds_bbox[3:], gts_bbox[3:]),
    }

    prec_rec = calculate_pr_curve(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=ious,
        preds_conf=preds_conf,
        frame_ind_dict_preds={0: (0, 3), 1: (3, 6)},
        frame_ind_dict_gts={0: (0, 3), 1: (3, 6)},
        area_range=(0, 100),
        iou_threshold=0.3,
    )

    npt.assert_array_equal(
        prec_rec,
        np.array(
            [
                [1, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6],
                [1 / 6, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6],
            ],
            dtype=np.float32,
        ),
    )


def test_coco_pr_curve_example_2():
    """Simple 2 frame example, one ignored GT and ignored unmatched pred"""
    preds_bbox = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 0.1, 1],
        ],
        dtype=np.float32,
    )
    gts_bbox = np.array(
        [
            [0.1, 0.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [2.1, 2.1, 1, 1],
            [0.1, 0.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [2.5, 2.5, 0.1, 1],
        ],
        dtype=np.float32,
    )
    preds_conf = np.array([0.7, 0.9, 0.8, 0.81, 0.6, 0.85], dtype=np.float32)
    ious = {
        0: 1 - iou_dist(preds_bbox[:3], gts_bbox[:3]),
        1: 1 - iou_dist(preds_bbox[3:], gts_bbox[3:]),
    }

    prec_rec = calculate_pr_curve(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=ious,
        preds_conf=preds_conf,
        frame_ind_dict_preds={0: (0, 3), 1: (3, 6)},
        frame_ind_dict_gts={0: (0, 3), 1: (3, 6)},
        area_range=(0.5, 100),
        iou_threshold=0.3,
    )

    npt.assert_array_equal(
        prec_rec,
        np.array(
            [[1, 1, 1, 1, 1], [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize(
    "iou_threshold,exp_result",
    (
        (
            0.5,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 0.8889, 0.9, 0.9091, 0.9167],
                    [
                        0.0833,
                        0.1667,
                        0.25,
                        0.3333,
                        0.4167,
                        0.5,
                        0.5833,
                        2 / 3,
                        2 / 3,
                        0.75,
                        0.8333,
                        0.9167,
                    ],
                ]
            ),
        ),
        (
            0.75,
            np.array(
                [
                    [
                        1,
                        0.5,
                        2 / 3,
                        0.5,
                        0.6,
                        2 / 3,
                        0.7143,
                        0.75,
                        2 / 3,
                        0.7,
                        0.6364,
                        2 / 3,
                    ],
                    [
                        0.0833,
                        0.0833,
                        0.1667,
                        0.1667,
                        0.25,
                        1 / 3,
                        0.4167,
                        0.5,
                        0.5,
                        0.5833,
                        0.5833,
                        2 / 3,
                    ],
                ]
            ),
        ),
    ),
)
def test_coco_pr_curve_example_3(iou_threshold: float, exp_result: np.ndarray):
    """
    Inspired by example here
    https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/README.md#a-practical-example
    """
    gts_bbox = np.array(
        [
            [0, 0, 1, 1],
            [0.5, 0.5, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0.5, 0.5, 1, 1],
            [0, 0, 1, 1],
            [1.1, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    preds_bbox = np.array(
        [
            [0, 0, 1, 1],
            [0.5, 0.5, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1.2, 1.2],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1.2, 1.2],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1.2, 1.2],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    preds_conf = np.array(
        [0.89, 0.82, 0.96, 0.99, 0.81, 0.86, 0.76, 0.95, 0.92, 0.85, 0.98, 0.94],
        dtype=np.float32,
    )
    frame_ind_dict_preds = {
        0: (0, 2),
        1: (2, 3),
        2: (3, 4),
        3: (4, 5),
        4: (5, 7),
        5: (7, 8),
        6: (8, 9),
        7: (9, 10),
        8: (10, 11),
        9: (11, 12),
    }
    frame_ind_dict_gts = {
        0: (0, 2),
        1: (2, 3),
        2: (3, 4),
        3: (4, 6),
        4: (6, 8),
        5: (8, 9),
        6: (9, 10),
        8: (10, 11),
        9: (11, 12),
    }

    ious = {}
    for k in set(frame_ind_dict_preds.keys()).union(frame_ind_dict_gts.keys()):
        gt_inds = frame_ind_dict_gts.get(k, (0, 0))
        preds_inds = frame_ind_dict_preds.get(k, (0, 0))
        ious[k] = 1 - iou_dist(
            preds_bbox[preds_inds[0] : preds_inds[1]], gts_bbox[gt_inds[0] : gt_inds[1]]
        )

    prec_rec = calculate_pr_curve(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=ious,
        preds_conf=preds_conf,
        frame_ind_dict_preds=frame_ind_dict_preds,
        frame_ind_dict_gts=frame_ind_dict_gts,
        area_range=(0, 100),
        iou_threshold=iou_threshold,
    )

    npt.assert_array_almost_equal(prec_rec, exp_result, 4)
