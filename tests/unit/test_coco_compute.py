import numba
import numpy as np
import numpy.testing as npt
import pytest

from evaldet.det.coco import evaluate_dataset, evaluate_image
from evaldet.dist import iou_dist


def _ious_to_numba(dious: dict[int, np.ndarray]) -> numba.typed.Dict:
    ious = numba.typed.Dict.empty(
        key_type=numba.int64, value_type=numba.float32[:, ::1]
    )
    for key, val in dious.items():
        ious[key] = val

    return ious


def test_coco_evaluate_image_both_empty() -> None:
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


def test_coco_evaluate_image_preds_empty() -> None:
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


def test_coco_evaluate_image_gts_empty() -> None:
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


def test_coco_evaluate_image_gts_all_ignored() -> None:
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


def test_coco_evaluate_image_gts_small_ignored() -> None:
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


def test_coco_evaluate_image_gts_large_ignored() -> None:
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


def test_coco_evaluate_image_preds_ignored_matched() -> None:
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


def test_coco_evaluate_image_preds_unmatched_ignored() -> None:
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


def test_coco_evaluate_image_matching_1() -> None:
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


def test_coco_evaluate_image_matching_2() -> None:
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


def test_coco_evaluate_image_matching_3() -> None:
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


def test_coco_evaluate_dataset_no_gts() -> None:
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.random.rand(1, 4).astype(np.float32),
        gts_bbox=np.zeros((0, 4), dtype=np.float32),
        ious=_ious_to_numba({0: np.zeros((1, 0), dtype=np.float32)}),
        preds_conf=np.array([0.5], dtype=np.float32),
        img_ind_corr=np.array([[0, 1, 0, 0]], dtype=np.int32),
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.full((1,), -1, dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.zeros((1,), dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.zeros((0,), dtype=np.bool_))


def test_coco_evaluate_dataset_gts_all_ignored() -> None:
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 0.1, 1]], dtype=np.float32),
        ious=_ious_to_numba({0: np.array([[0.1]], dtype=np.float32)}),
        preds_conf=np.array([0.5], dtype=np.float32),
        img_ind_corr=np.array([[0, 1, 0, 1]], dtype=np.int32),
        area_range=(0.5, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.full((1,), -1, dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.zeros((1,), dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.ones((1,), dtype=np.bool_))


def test_coco_evaluate_dataset_no_preds() -> None:
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.zeros((0, 4), dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious=_ious_to_numba({0: np.zeros((0, 1), dtype=np.float32)}),
        preds_conf=np.zeros((0,), dtype=np.float32),
        img_ind_corr=np.array([[0, 0, 0, 1]], dtype=np.int32),
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.zeros((0,), dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.zeros((0,), dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.zeros((1,), dtype=np.bool_))


def test_coco_evaluate_dataset_preds_all_ignored() -> None:
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.array([[0, 0, 0.1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious=_ious_to_numba({0: np.array([[0.1]], dtype=np.float32)}),
        preds_conf=np.array([0.5], dtype=np.float32),
        img_ind_corr=np.array([[0, 1, 0, 1]], dtype=np.int32),
        area_range=(0.5, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.full((1,), -1, dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.ones((1,), dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.zeros((1,), dtype=np.bool_))


def test_coco_evaluate_dataset_no_preds_image() -> None:
    """Preds missing on one image"""
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.array(
            [
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        gts_bbox=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        ious=_ious_to_numba(
            {
                0: np.array([[1]], dtype=np.float32),
                1: np.zeros((0, 1), dtype=np.float32),
            }
        ),
        preds_conf=np.array([0.5], dtype=np.float32),
        img_ind_corr=np.array([[0, 1, 0, 1], [0, 0, 1, 2]], dtype=np.int32),
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.array([0], dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.array([0], dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.array([0, 0], dtype=np.bool_))


def test_coco_evaluate_dataset_no_gts_image() -> None:
    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        gts_bbox=np.array([[0, 0, 1, 1]], dtype=np.float32),
        ious=_ious_to_numba(
            {
                0: np.array([[1]], dtype=np.float32),
                1: np.zeros((1, 0), dtype=np.float32),
            }
        ),
        preds_conf=np.array([0.5, 0.6], dtype=np.float32),
        img_ind_corr=np.array([[0, 1, 0, 1], [1, 2, 0, 0]], dtype=np.int32),
        area_range=(0, 100),
        iou_threshold=0.5,
    )

    npt.assert_array_equal(det_matched, np.array([0, -1], dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.array([0, 0], dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.array([0], dtype=np.bool_))


def test_coco_evaluate_dataset_example_1() -> None:
    """Simple 2 image example, one unmatched GT and one unmatched pred"""
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
            [2.1, 2.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [1.1, 1.1, 1, 1],
            [0.1, 0.1, 1, 1],
            [2.5, 2.5, 1, 1],
        ],
        dtype=np.float32,
    )
    preds_conf = np.array([0.7, 0.9, 0.8, 0.81, 0.6, 0.85], dtype=np.float32)
    ious = {
        0: 1 - iou_dist(preds_bbox[:3], gts_bbox[:3]),
        1: 1 - iou_dist(preds_bbox[3:], gts_bbox[3:]),
    }

    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=_ious_to_numba(ious),
        preds_conf=preds_conf,
        img_ind_corr=np.array([[0, 3, 0, 3], [3, 6, 3, 6]], dtype=np.int32),
        area_range=(0, 100),
        iou_threshold=0.3,
    )

    npt.assert_array_equal(det_matched, np.array([0, 2, 1, 4, 3, -1], dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.array([0, 0, 0, 0, 0, 0], dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.array([0, 0, 0, 0, 0, 0], dtype=np.bool_))


def test_coco_evaluate_dataset_example_2() -> None:
    """Simple 2 image example, one ignored GT and ignored unmatched pred"""
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
            [2.5, 2.5, 0.1, 1],
            [0.1, 0.1, 1, 1],
            [1.1, 1.1, 1, 1],
        ],
        dtype=np.float32,
    )
    preds_conf = np.array([0.7, 0.9, 0.8, 0.81, 0.6, 0.85], dtype=np.float32)
    ious = {
        0: 1 - iou_dist(preds_bbox[:3], gts_bbox[:3]),
        1: 1 - iou_dist(preds_bbox[3:], gts_bbox[3:]),
    }

    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=_ious_to_numba(ious),
        preds_conf=preds_conf,
        img_ind_corr=np.array([[0, 3, 0, 3], [3, 6, 3, 6]], dtype=np.int32),
        area_range=(0.5, 100),
        iou_threshold=0.3,
    )

    npt.assert_array_equal(det_matched, np.array([0, 1, 2, 4, 5, -1], dtype=np.int32))
    npt.assert_array_equal(det_ignored, np.array([0, 0, 0, 0, 0, 1], dtype=np.bool_))
    npt.assert_array_equal(gts_ignored, np.array([0, 0, 0, 1, 0, 0], dtype=np.bool_))


@pytest.mark.parametrize(
    "iou_threshold,exp_match",
    (
        (0.5, np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, -1, 10, 11], dtype=np.int32)),
        (0.75, np.array([0, 1, 2, 3, -1, 6, 7, -1, 9, -1, -1, 11], dtype=np.int32)),
    ),
)
def test_coco_evaluate_dataset_example_3(
    iou_threshold: float, exp_match: np.ndarray
) -> None:
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
    img_ind_corr = np.array(
        [
            [0, 2, 0, 2],
            [2, 3, 2, 3],
            [3, 4, 3, 4],
            [4, 5, 4, 6],
            [5, 7, 6, 8],
            [7, 8, 8, 9],
            [8, 9, 9, 10],
            [9, 10, 0, 0],
            [10, 11, 10, 11],
            [11, 12, 11, 12],
        ],
        dtype=np.int32,
    )

    ious = {}
    for i in range(len(img_ind_corr)):
        gt_inds = img_ind_corr[i, 2:]
        preds_inds = img_ind_corr[i, :2]
        ious[i] = 1 - iou_dist(
            preds_bbox[preds_inds[0] : preds_inds[1]], gts_bbox[gt_inds[0] : gt_inds[1]]
        )

    det_matched, det_ignored, gts_ignored = evaluate_dataset(
        preds_bbox=preds_bbox,
        gts_bbox=gts_bbox,
        ious=_ious_to_numba(ious),
        preds_conf=preds_conf,
        img_ind_corr=img_ind_corr,
        area_range=(0, 100),
        iou_threshold=iou_threshold,
    )

    npt.assert_array_equal(det_matched, exp_match)
    npt.assert_array_equal(
        det_ignored, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    )
    npt.assert_array_equal(
        gts_ignored, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    )
