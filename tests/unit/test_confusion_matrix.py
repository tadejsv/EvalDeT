import numpy as np
import numpy.testing as npt

from evaldet.utils.confusion_matrix import confusion_matrix


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
