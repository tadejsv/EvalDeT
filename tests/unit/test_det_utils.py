import numpy as np
import numpy.testing as npt
import pytest

from evaldet.det.utils import compute_ious, confusion_matrix, match_images
from evaldet.detections import Detections


@pytest.fixture
def empty_dets() -> Detections:
    return Detections([], [], [], confs=[], class_names=("cls",), image_names=())


def test_confusion_matrix_empty_gts_and_preds1() -> None:
    cm = confusion_matrix(
        matching=np.zeros((0,), dtype=np.int32),
        gt_ignored=np.zeros((0,), dtype=np.bool_),
        hyp_classes=np.zeros((0,), dtype=np.int32),
        gt_classes=np.zeros((0,), dtype=np.int32),
        n_classes=0,
    )

    npt.assert_array_equal(cm, np.zeros((1, 1), dtype=np.int32))


def test_confusion_matrix_empty_gts_and_preds2() -> None:
    cm = confusion_matrix(
        matching=np.zeros((0,), dtype=np.int32),
        gt_ignored=np.zeros((0,), dtype=np.bool_),
        hyp_classes=np.zeros((0,), dtype=np.int32),
        gt_classes=np.zeros((0,), dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.zeros((2, 2), dtype=np.int32))


def test_confusion_matrix_empty_gts() -> None:
    cm = confusion_matrix(
        matching=np.array([-1], dtype=np.int32),
        gt_ignored=np.zeros((0,), dtype=np.bool_),
        hyp_classes=np.array([0], dtype=np.int32),
        gt_classes=np.zeros((0,), dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_confusion_matrix_empty_preds() -> None:
    cm = confusion_matrix(
        matching=np.zeros((0,), dtype=np.int32),
        gt_ignored=np.array([False], dtype=np.bool_),
        hyp_classes=np.zeros((0,), dtype=np.int32),
        gt_classes=np.array([0], dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_confusion_matrix_all_ignored_gts() -> None:
    cm = confusion_matrix(
        matching=np.array([-1], dtype=np.int32),
        gt_ignored=np.array([True], dtype=np.bool_),
        hyp_classes=np.array([0], dtype=np.int32),
        gt_classes=np.array([0], dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_confusion_matrix_all_ignored_gts_matched() -> None:
    cm = confusion_matrix(
        matching=np.array([0], dtype=np.int32),
        gt_ignored=np.array([True], dtype=np.bool_),
        hyp_classes=np.array([0], dtype=np.int32),
        gt_classes=np.array([0], dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_confusion_matrix_no_matching() -> None:
    cm = confusion_matrix(
        matching=np.array([-1], dtype=np.int32),
        gt_ignored=np.array([False], dtype=np.bool_),
        hyp_classes=np.array([0], dtype=np.int32),
        gt_classes=np.array([0], dtype=np.int32),
        n_classes=1,
    )

    npt.assert_array_equal(cm, np.array([[0, 1], [1, 0]], dtype=np.int32))


def test_confusion_matrix_normal_1() -> None:
    """Normal matching test, all matched"""

    cm = confusion_matrix(
        matching=np.array([4, 2, 1, 0, 3], dtype=np.int32),
        gt_ignored=np.array([False, False, False, False, False], dtype=np.bool_),
        hyp_classes=np.array([0, 0, 1, 2, 2], dtype=np.int32),
        gt_classes=np.array([0, 2, 2, 1, 0], dtype=np.int32),
        n_classes=3,
    )

    npt.assert_array_equal(
        cm,
        np.array(
            [[1, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        ),
    )


def test_confusion_matrix_normal_2() -> None:
    """Normal matching test, some ignored and non-matched"""
    cm = confusion_matrix(
        matching=np.array([4, 2, 1, -1, 3], dtype=np.int32),
        gt_ignored=np.array([False, False, False, True, False], dtype=np.bool_),
        hyp_classes=np.array([0, 0, 1, 2, 2], dtype=np.int32),
        gt_classes=np.array([0, 2, 2, 1, 0], dtype=np.int32),
        n_classes=4,
    )

    npt.assert_array_equal(
        cm,
        np.array(
            [
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_compute_ious_empty() -> None:
    ious = compute_ious(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.int32),
    )
    assert len(ious) == 0


def test_compute_ious_no_hyp() -> None:
    ious = compute_ious(
        np.zeros((0, 4), dtype=np.float32),
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 0, 0, 1]], dtype=np.int32),
    )
    assert len(ious) == 0


def test_compute_ious_no_gt() -> None:
    ious = compute_ious(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.array([[0, 1, 0, 0]], dtype=np.int32),
    )
    assert len(ious) == 0


def test_compute_ious_one_image_simple() -> None:
    ious = compute_ious(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 1, 0, 1]], dtype=np.int32),
    )
    assert len(ious) == 1

    npt.assert_array_equal(ious[0], np.array([[1.0]], dtype=np.float32))


def test_compute_ious_two_images_missing_gt() -> None:
    ious = compute_ious(
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 1, 0, 1], [1, 2, 0, 0]], dtype=np.int32),
    )
    assert len(ious) == 1

    npt.assert_array_equal(ious[0], np.array([[1.0]], dtype=np.float32))


def test_compute_ious_two_images_missing_hyp() -> None:
    ious = compute_ious(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 1, 0, 1], [0, 0, 1, 2]], dtype=np.int32),
    )
    assert len(ious) == 1

    npt.assert_array_equal(ious[0], np.array([[1.0]], dtype=np.float32))


def test_compute_ious_two_images_simple() -> None:
    ious = compute_ious(
        np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0.5, 0.5, 1, 1]], dtype=np.float32),
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        np.array([[0, 1, 0, 1], [1, 3, 1, 2]], dtype=np.int32),
    )
    assert len(ious) == 2

    npt.assert_array_equal(ious[0], np.array([[1.0]], dtype=np.float32))
    npt.assert_array_almost_equal(ious[1], np.array([[1.0], [1 / 7]], dtype=np.float32))


def test_match_images_empty(empty_dets: Detections) -> None:
    res = match_images(empty_dets, empty_dets)
    npt.assert_array_equal(res, np.zeros((0, 4), dtype=np.int32))


def test_match_images_empty_hyp(empty_dets: Detections) -> None:
    gt = Detections([0], np.array([[0, 0, 1, 1]]), [0], ("cls",), ("im",))
    res = match_images(gt, empty_dets)
    npt.assert_array_equal(res, np.array([[0, 0, 0, 1]], dtype=np.int32))


def test_match_images_empty_gt(empty_dets: Detections) -> None:
    hyp = Detections([0], np.array([[0, 0, 1, 1]]), [0], ("cls",), ("im",), [0.1])
    res = match_images(empty_dets, hyp)
    npt.assert_array_equal(res, np.array([[0, 1, 0, 0]], dtype=np.int32))


def test_match_images_all_matching() -> None:
    hyp = Detections([0], np.array([[0, 0, 1, 1]]), [0], ("cls",), ("im",), [0.1])
    gt = Detections([0], np.array([[0, 0, 1, 1]]), [0], ("cls",), ("im",), None)
    res = match_images(gt, hyp)

    npt.assert_array_equal(res, np.array([[0, 1, 0, 1]], dtype=np.int32))


def test_match_images_missing() -> None:
    hyp = Detections(
        [0, 1],
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]]),
        [0, 0],
        ("cls",),
        ("im1", "im2"),
        [0.1, 0.1],
    )
    gt = Detections(
        [0, 1],
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]]),
        [0, 0],
        ("cls",),
        ("im1", "im3"),
        None,
    )
    res = match_images(gt, hyp)

    npt.assert_array_equal(
        res, np.array([[0, 1, 0, 1], [1, 2, 0, 0], [0, 0, 1, 2]], dtype=np.int32)
    )
