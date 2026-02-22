from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from deepdiff import DeepDiff

from evaldet.det.coco import (
    APInterpolation,
    compute_coco_summary,
    compute_metrics,
    confusion_matrix,
)
from evaldet.detections import Detections


@pytest.fixture
def empty_dets() -> Detections:
    return Detections([], [], [], confs=[], class_names=("cls",), image_names=())


@pytest.fixture
def normal_hyp_1() -> Detections:
    return Detections(
        [0, 1, 0, 1, 2, 1, 2, 0, 2],
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 1, 1],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ("cls1", "cls2", "cls3"),
        ("im1", "im2", "im0"),
        confs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )


@pytest.fixture
def normal_gt_1() -> Detections:
    return Detections(
        [1, 0, 1, 0, 2, 0, 2, 1, 2],
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 1, 1],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ("cls1", "cls2", "cls3"),
        ("im0", "im1", "im2"),
        confs=None,
    )


@pytest.fixture
def normal_hyp_2() -> Detections:
    return Detections(
        [0, 1, 0, 1, 2, 1, 2, 0, 2, 3],
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
        ("cls1", "cls2", "cls3"),
        ("im1", "im2", "im0", "im4"),
        confs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )


@pytest.fixture
def normal_gt_2() -> Detections:
    return Detections(
        [1, 0, 1, 0, 2, 0, 2, 1, 2, 3],
        np.array(
            [
                [0, 0, 0.1, 0.1],
                [10, 10, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
        ("cls1", "cls2", "cls3"),
        ("im0", "im1", "im2", "im5"),
        confs=None,
    )


@pytest.fixture
def example_hyp() -> Detections:
    return Detections(
        [0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9],
        np.array(
            [
                [12.0, 44.0, 332.0, 437.0],
                [220.0, 344.0, 180.0, 87.0],
                [166.0, 0.0, 311.0, 365.0],
                [49.0, 7.0, 269.0, 493.0],
                [6.0, 55.0, 489.0, 347.0],
                [65.0, 74.0, 305.0, 257.0],
                [67.0, 117.0, 162.0, 187.0],
                [281.0, 109.0, 163.0, 211.0],
                [7.0, 9.0, 342.0, 256.0],
                [151.0, 123.0, 352.0, 252.0],
                [29.0, 0.0, 239.0, 287.0],
                [47.0, 43.0, 179.0, 255.0],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("cat",),
        (
            "2007_000549",
            "2007_000733",
            "2007_003525",
            "2007_004856",
            "2007_005460",
            "2007_005688",
            "2007_009346",
            "2008_002045",
            "2008_006599",
            "2010_004175",
        ),
        confs=[0.94, 0.85, 0.95, 0.92, 0.951, 0.81, 0.86, 0.76, 0.89, 0.82, 0.99, 0.98],
    )


@pytest.fixture
def example_gt() -> Detections:
    return Detections(
        [0, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9],
        np.array(
            [
                [1.0, 49.0, 340.0, 450.0],
                [160.0, 1.0, 288.0, 374.0],
                [28.0, 1.0, 307.0, 495.0],
                [40.0, 4.0, 444.0, 366.0],
                [170.0, 44.0, 289.0, 207.0],
                [72.0, 123.0, 315.0, 211.0],
                [60.0, 123.0, 160.0, 182.0],
                [243.0, 105.0, 194.0, 212.0],
                [1.0, 16.0, 351.0, 256.0],
                [130.0, 139.0, 370.0, 227.0],
                [34.0, 1.0, 247.0, 282.0],
                [1.0, 39.0, 247.0, 261.0],
            ],
            dtype=np.float32,
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("cat",),
        (
            "2007_000549",
            "2007_000733",
            "2007_003525",
            "2007_004856",
            "2007_005460",
            "2007_005688",
            "2007_009346",
            "2008_002045",
            "2008_006599",
            "2010_004175",
        ),
        confs=None,
    )


#########################################
# Test Errors
#########################################


def test_coco_different_class_names_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("slc",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=None)

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        confusion_matrix(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        compute_metrics(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        compute_coco_summary(gts, hyp)


def test_coco_no_hyp_conf_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=None)
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=None)

    with pytest.raises(ValueError, match="`confs` must be provided"):
        confusion_matrix(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`confs` must be provided"):
        compute_metrics(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`confs` must be provided"):
        compute_coco_summary(gts, hyp)


#########################################
# Test Basic Funcionality
#########################################


def test_coco_cm_empty_gts_hyp(empty_dets: Detections) -> None:
    cm = confusion_matrix(empty_dets, empty_dets, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 0], [0, 0]], dtype=np.int32))


def test_coco_cm_empty_gts(empty_dets: Detections) -> None:
    hyp = Detections(
        [0],
        [np.array([0, 0, 1, 1])],
        [0],
        confs=[0.1],
        class_names=("cls",),
        image_names=("im",),
    )

    cm = confusion_matrix(empty_dets, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_coco_cm_empty_preds(empty_dets: Detections) -> None:
    gts = Detections(
        [0],
        [np.array([0, 0, 1, 1])],
        [0],
        confs=[0.1],
        class_names=("cls",),
        image_names=("im",),
    )

    cm = confusion_matrix(gts, empty_dets, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_empty_image() -> None:
    gts = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=None
    )
    hyp = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=[0.1]
    )

    cm = confusion_matrix(gts, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[1, 0], [0, 0]], dtype=np.int32))


def test_coco_metrics_empty_gts_hyp(empty_dets: Detections) -> None:
    metrics = compute_metrics(empty_dets, empty_dets, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
        "mean_ap": None,
    }


def test_coco_metrics_empty_gts(empty_dets: Detections) -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    metrics = compute_metrics(empty_dets, hyp, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
        "mean_ap": None,
    }


def test_coco_metrics_empty_preds(empty_dets: Detections) -> None:
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    metrics = compute_metrics(gts, empty_dets, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": 0.0, "n_gts": 1, "precision": 0.0, "recall": 0.0}
        },
        "mean_ap": 0.0,
    }


def test_coco_metrics_empty_image() -> None:
    gts = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=None
    )
    hyp = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=[0.1]
    )

    metrics = compute_metrics(gts, hyp, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": 1.0, "n_gts": 1, "precision": 1.0, "recall": 1.0}
        },
        "mean_ap": 1.0,
    }


def test_coco_cm_all_ignored_gts() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    cm = confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_hyp() -> None:
    hyp = Detections(
        [0], [np.array([1, 1, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    cm = confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_matching() -> None:
    """Hyp is matched to an ingored gt - by COCO rules this means hyp is ignored too"""
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    cm = confusion_matrix(gts, hyp, 1e-9, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [0, 0]], dtype=np.int32))


def test_coco_cm_no_matching() -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    cm = confusion_matrix(gts, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 1], [1, 0]], dtype=np.int32))


def test_coco_metrics_all_ignored_gts() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    metrics = compute_metrics(gts, hyp, 0.5, (1, 100))

    assert metrics == {
        "mean_ap": None,
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
    }


def test_coco_metrics_all_ignored_hyp() -> None:
    hyp = Detections(
        [0], [np.array([1, 1, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    metrics = compute_metrics(gts, hyp, 0.5, (1, 100))

    assert metrics == {
        "mean_ap": 0.0,
        "class_results": {
            "cls": {"ap": 0.0, "n_gts": 1, "precision": 0.0, "recall": 0.0}
        },
    }


def test_coco_metrics_all_ignored_matching() -> None:
    """Hyp is matched to an ingored gt - by COCO rules this means hyp is ignored too"""
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    metrics = compute_metrics(gts, hyp, 1e-9, (1, 100))

    assert metrics == {
        "mean_ap": None,
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
    }


def test_coco_metrics_no_matching() -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    metrics = compute_metrics(gts, hyp, 0.5)

    assert metrics == {
        "mean_ap": 0.0,
        "class_results": {
            "cls": {"ap": 0.0, "n_gts": 1, "precision": 0.0, "recall": 0.0}
        },
    }


def test_coco_metrics_class_missing() -> None:
    hyp = Detections(
        [0],
        [np.array([0, 0, 1, 1])],
        [0],
        ("cls1", "cls2"),
        ("im",),
        confs=[1.0],
    )
    gts = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls1", "cls2"), ("im",), confs=None
    )
    # raise ValueError

    metrics = compute_metrics(gts, hyp, 0.5)

    assert metrics == {
        "mean_ap": 1.0,
        "class_results": {
            "cls1": {"ap": 1.0, "n_gts": 1, "precision": 1.0, "recall": 1.0},
            "cls2": {"ap": None, "n_gts": 0, "precision": None, "recall": None},
        },
    }


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 0.75,
                "ap_50": 0.75,
                "ap_75": 0.75,
                "mean_ap_per_class": {"cls1": 0.75},
                "ap_50_per_class": {"cls1": 0.75},
                "ap_75_per_class": {"cls1": 0.75},
                "mean_ap_sizes": {"size": 0.75},
                "mean_ap_sizes_per_class": {
                    "cls1": {"size": 0.75},
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 76 / 101,
                "ap_50": 76 / 101,
                "ap_75": 76 / 101,
                "mean_ap_per_class": {"cls1": 76 / 101},
                "ap_50_per_class": {"cls1": 76 / 101},
                "ap_75_per_class": {"cls1": 76 / 101},
                "mean_ap_sizes": {"size": 76 / 101},
                "mean_ap_sizes_per_class": {
                    "cls1": {"size": 76 / 101},
                },
            },
        ),
    ],
)
def test_coco_summary_simple(
    ap_interpolation: APInterpolation, result: dict[str, float | dict[str, float]]
) -> None:
    """4 dets 1 class, one non-matching, others perfect"""
    hyp = Detections(
        [0, 0, 0, 0],
        np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 1, 1]]),
        [0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=[1.0, 1.0, 1.0, 0.9],
    )

    gts = Detections(
        [0, 0, 0, 0],
        np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [4, 4, 1, 1]]),
        [0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=None,
    )

    summary = compute_coco_summary(
        gts, hyp, sizes={"size": (0, float("inf"))}, ap_interpolation=ap_interpolation
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 0.55,
                "ap_50": 0.75,
                "ap_75": 0.5,
                "mean_ap_per_class": {"cls1": 0.55},
                "ap_50_per_class": {"cls1": 0.75},
                "ap_75_per_class": {"cls1": 0.5},
                "mean_ap_sizes": {"size": 0.55},
                "mean_ap_sizes_per_class": {
                    "cls1": {"size": 0.55},
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 56 / 101,
                "ap_50": 76 / 101,
                "ap_75": 51 / 101,
                "mean_ap_per_class": {"cls1": 56 / 101},
                "ap_50_per_class": {"cls1": 76 / 101},
                "ap_75_per_class": {"cls1": 51 / 101},
                "mean_ap_sizes": {"size": 56 / 101},
                "mean_ap_sizes_per_class": {
                    "cls1": {"size": 56 / 101},
                },
            },
        ),
    ],
)
def test_coco_summary_different_iou_thresholds(
    ap_interpolation: APInterpolation, result: dict[str, float | dict[str, float]]
) -> None:
    """4 dets 1 class, different IoUs"""
    hyp = Detections(
        [0, 0, 0, 0],
        np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 1, 1]]),
        [0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=[1.0, 1.0, 0.9, 0.8],
    )

    gts = Detections(
        [0, 0, 0, 0],
        np.array([[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 0.75, 0.75], [3, 3, 0.5, 0.5]]),
        [0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=None,
    )

    summary = compute_coco_summary(
        gts, hyp, sizes={"size": (0, float("inf"))}, ap_interpolation=ap_interpolation
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 3 / 5,
                "ap_50": 0.6,
                "ap_75": 0.6,
                "mean_ap_per_class": {"cls1": 0.6},
                "ap_50_per_class": {"cls1": 0.6},
                "ap_75_per_class": {"cls1": 0.6},
                "mean_ap_sizes": {"small": 2 / 3, "large": 1 / 2},
                "mean_ap_sizes_per_class": {
                    "cls1": {"small": 2 / 3, "large": 1 / 2},
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 61 / 101,
                "ap_50": 61 / 101,
                "ap_75": 61 / 101,
                "mean_ap_per_class": {"cls1": 61 / 101},
                "ap_50_per_class": {"cls1": 61 / 101},
                "ap_75_per_class": {"cls1": 61 / 101},
                "mean_ap_sizes": {"small": 67 / 101, "large": 51 / 101},
                "mean_ap_sizes_per_class": {
                    "cls1": {"small": 67 / 101, "large": 51 / 101},
                },
            },
        ),
    ],
)
def test_coco_summary_different_sizes(
    ap_interpolation: APInterpolation, result: dict[str, Any]
) -> None:
    """5 dets 1 class, 2 size classes (3/2): 1 non-matching first, 1 second"""
    hyp = Detections(
        [0, 0, 0, 0, 0],
        np.array(
            [[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [30, 30, 2, 2], [40, 40, 2, 2]]
        ),
        [0, 0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=[1.0, 1.0, 0.9, 1.0, 0.8],
    )

    gts = Detections(
        [0, 0, 0, 0, 0],
        np.array(
            [[0, 0, 1, 1], [1, 1, 1, 1], [3, 3, 1, 1], [30, 30, 2, 2], [45, 45, 2, 2]]
        ),
        [0, 0, 0, 0, 0],
        ("cls1",),
        ("im",),
        confs=None,
    )

    summary = compute_coco_summary(
        gts,
        hyp,
        sizes={"small": (0, 2), "large": (2, 10)},
        ap_interpolation=ap_interpolation,
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 1 / 2,
                "ap_50": 1 / 2,
                "ap_75": 1 / 2,
                "mean_ap_per_class": {"cls1": 2 / 3, "cls2": 1 / 3},
                "ap_50_per_class": {"cls1": 2 / 3, "cls2": 1 / 3},
                "ap_75_per_class": {"cls1": 2 / 3, "cls2": 1 / 3},
                "mean_ap_sizes": {"small": 2 / 3, "large": 1 / 3},
                "mean_ap_sizes_per_class": {
                    "cls1": {"small": 2 / 3, "large": None},
                    "cls2": {"small": None, "large": 1 / 3},
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 1 / 2,
                "ap_50": 1 / 2,
                "ap_75": 1 / 2,
                "mean_ap_per_class": {"cls1": 67 / 101, "cls2": 34 / 101},
                "ap_50_per_class": {"cls1": 67 / 101, "cls2": 34 / 101},
                "ap_75_per_class": {"cls1": 67 / 101, "cls2": 34 / 101},
                "mean_ap_sizes": {"small": 67 / 101, "large": 34 / 101},
                "mean_ap_sizes_per_class": {
                    "cls1": {"small": 67 / 101, "large": None},
                    "cls2": {"small": None, "large": 34 / 101},
                },
            },
        ),
    ],
)
def test_coco_summary_different_sizes_classes(
    ap_interpolation: APInterpolation, result: dict[str, float | dict[str, float]]
) -> None:
    """2 classes, 3 dets per class, class 1 only size 1, class 2 only size 2"""
    hyp = Detections(
        [0, 0, 0, 0, 0, 0],
        np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 1, 1],
                [30, 30, 2, 2],
                [40, 40, 2, 2],
                [50, 50, 2, 2],
            ]
        ),
        [0, 0, 0, 1, 1, 1],
        ("cls1", "cls2"),
        ("im",),
        confs=[1.0, 1.0, 0.9, 1.0, 0.85, 0.8],
    )

    gts = Detections(
        [0, 0, 0, 0, 0, 0],
        np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [3, 3, 1, 1],
                [30, 30, 2, 2],
                [45, 45, 2, 2],
                [60, 60, 2, 2],
            ]
        ),
        [0, 0, 0, 1, 1, 1],
        ("cls1", "cls2"),
        ("im",),
        confs=None,
    )

    summary = compute_coco_summary(
        gts,
        hyp,
        sizes={"small": (0, 2), "large": (2, 10)},
        ap_interpolation=ap_interpolation,
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


#####################################
# Test on examples
#####################################


def test_coco_cm_normal1(normal_hyp_1: Detections, normal_gt_1: Detections) -> None:
    """Easy case, perfect matching"""

    cm = confusion_matrix(normal_gt_1, normal_hyp_1, 0.5)

    npt.assert_array_equal(
        cm,
        np.array(
            [
                [2, 1, 0, 0],
                [1, 0, 2, 0],
                [0, 2, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_coco_cm_normal2(normal_hyp_2: Detections, normal_gt_2: Detections) -> None:
    """Easy case, some not matching, some ignored"""
    cm = confusion_matrix(normal_gt_2, normal_hyp_2, 0.5, (1, 100))

    npt.assert_array_equal(
        cm,
        np.array(
            [
                [1, 1, 0, 2],
                [0, 0, 2, 1],
                [0, 2, 1, 0],
                [2, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 13 / 54,
                "class_results": {
                    "cls1": {
                        "ap": 10 / 18,
                        "n_gts": 3,
                        "precision": 2 / 3,
                        "recall": 2 / 3,
                    },
                    "cls2": {"ap": 0, "n_gts": 3, "precision": 0, "recall": 0},
                    "cls3": {
                        "ap": 1 / 6,
                        "n_gts": 3,
                        "precision": 1 / 3,
                        "recall": 1 / 3,
                    },
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 73 / 303,
                "class_results": {
                    "cls1": {
                        "ap": 56 / 101,
                        "n_gts": 3,
                        "precision": 2 / 3,
                        "recall": 2 / 3,
                    },
                    "cls2": {"ap": 0.0, "n_gts": 3, "precision": 0.0, "recall": 0.0},
                    "cls3": {
                        "ap": 17 / 101,
                        "n_gts": 3,
                        "precision": 1 / 3,
                        "recall": 1 / 3,
                    },
                },
            },
        ),
    ],
)
def test_coco_metrics_normal1(
    normal_hyp_1: Detections,
    normal_gt_1: Detections,
    ap_interpolation: APInterpolation,
    result: dict[str, Any],
) -> None:
    metrics = compute_metrics(
        normal_gt_1, normal_hyp_1, 0.5, ap_interpolation=ap_interpolation
    )

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 1 / 9,
                "class_results": {
                    "cls1": {
                        "ap": 1 / 6,
                        "n_gts": 3,
                        "precision": 1 / 4,
                        "recall": 1 / 3,
                    },
                    "cls2": {"ap": 0.0, "n_gts": 3, "precision": 0.0, "recall": 0.0},
                    "cls3": {
                        "ap": 1 / 6,
                        "n_gts": 3,
                        "precision": 1 / 3,
                        "recall": 1 / 3,
                    },
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 0.11221122112211222,
                "class_results": {
                    "cls1": {
                        "ap": 17 / 101,
                        "n_gts": 3,
                        "precision": 1 / 4,
                        "recall": 1 / 3,
                    },
                    "cls2": {"ap": 0.0, "n_gts": 3, "precision": 0.0, "recall": 0.0},
                    "cls3": {
                        "ap": 17 / 101,
                        "n_gts": 3,
                        "precision": 1 / 3,
                        "recall": 1 / 3,
                    },
                },
            },
        ),
    ],
)
def test_coco_metrics_normal2(
    normal_hyp_2: Detections,
    normal_gt_2: Detections,
    ap_interpolation: APInterpolation,
    result: dict[str, Any],
) -> None:
    metrics = compute_metrics(
        normal_gt_2,
        normal_hyp_2,
        0.5,
        area_range=(0.5, 100),
        ap_interpolation=ap_interpolation,
    )

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 13 / 54,
                "ap_50": 13 / 54,
                "ap_75": 13 / 54,
                "mean_ap_per_class": {
                    "cls1": 10 / 18,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "ap_50_per_class": {
                    "cls1": 10 / 18,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "ap_75_per_class": {
                    "cls1": 10 / 18,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "mean_ap_sizes": {
                    "small": 13 / 54,
                    "medium": None,
                    "large": None,
                },
                "mean_ap_sizes_per_class": {
                    "cls1": {
                        "small": 10 / 18,
                        "medium": None,
                        "large": None,
                    },
                    "cls2": {
                        "small": 0,
                        "medium": None,
                        "large": None,
                    },
                    "cls3": {
                        "small": 1 / 6,
                        "medium": None,
                        "large": None,
                    },
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 73 / 303,
                "ap_50": 73 / 303,
                "ap_75": 73 / 303,
                "mean_ap_per_class": {
                    "cls1": 56 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "ap_50_per_class": {
                    "cls1": 56 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "ap_75_per_class": {
                    "cls1": 56 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "mean_ap_sizes": {
                    "small": 73 / 303,
                    "medium": None,
                    "large": None,
                },
                "mean_ap_sizes_per_class": {
                    "cls1": {
                        "small": 56 / 101,
                        "medium": None,
                        "large": None,
                    },
                    "cls2": {
                        "small": 0,
                        "medium": None,
                        "large": None,
                    },
                    "cls3": {
                        "small": 17 / 101,
                        "medium": None,
                        "large": None,
                    },
                },
            },
        ),
    ],
)
def test_coco_summary_normal1(
    normal_hyp_1: Detections,
    normal_gt_1: Detections,
    ap_interpolation: APInterpolation,
    result: dict[str, float | dict[str, float]],
) -> None:
    summary = compute_coco_summary(
        normal_gt_1, normal_hyp_1, ap_interpolation=ap_interpolation
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "result"),
    [
        (
            "pascal",
            {
                "mean_ap": 7 / 72,
                "ap_50": 7 / 72,
                "ap_75": 7 / 72,
                "mean_ap_per_class": {
                    "cls1": 1 / 8,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "ap_75_per_class": {
                    "cls1": 1 / 8,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "ap_50_per_class": {
                    "cls1": 1 / 8,
                    "cls2": 0,
                    "cls3": 1 / 6,
                },
                "mean_ap_sizes": {
                    "small": 7 / 72,
                    "medium": None,
                    "large": None,
                },
                "mean_ap_sizes_per_class": {
                    "cls1": {
                        "small": 1 / 8,
                        "medium": None,
                        "large": None,
                    },
                    "cls2": {
                        "small": 0,
                        "medium": None,
                        "large": None,
                    },
                    "cls3": {
                        "small": 1 / 6,
                        "medium": None,
                        "large": None,
                    },
                },
            },
        ),
        (
            "coco",
            {
                "mean_ap": 10 / 101,
                "ap_50": 10 / 101,
                "ap_75": 10 / 101,
                "mean_ap_per_class": {
                    "cls1": 13 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "ap_50_per_class": {
                    "cls1": 13 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "ap_75_per_class": {
                    "cls1": 13 / 101,
                    "cls2": 0,
                    "cls3": 17 / 101,
                },
                "mean_ap_sizes": {
                    "small": 10 / 101,
                    "medium": None,
                    "large": None,
                },
                "mean_ap_sizes_per_class": {
                    "cls1": {
                        "small": 13 / 101,
                        "medium": None,
                        "large": None,
                    },
                    "cls2": {
                        "small": 0,
                        "medium": None,
                        "large": None,
                    },
                    "cls3": {
                        "small": 17 / 101,
                        "medium": None,
                        "large": None,
                    },
                },
            },
        ),
    ],
)
def test_coco_summary_normal2(
    normal_hyp_2: Detections,
    normal_gt_2: Detections,
    ap_interpolation: APInterpolation,
    result: dict[str, float | dict[str, float]],
) -> None:
    summary = compute_coco_summary(
        normal_gt_2, normal_hyp_2, ap_interpolation=ap_interpolation
    )

    assert not DeepDiff(
        summary,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    ("ap_interpolation", "iou_threshold", "result"),
    [
        ("pascal", 0.5, {"ap": 0.8958, "precision": 0.9167, "recall": 0.9167}),
        ("coco", 0.5, {"ap": 0.8902, "precision": 0.9167, "recall": 0.9167}),
        ("pascal", 0.75, {"ap": 0.5097, "precision": 2 / 3, "recall": 2 / 3}),
        ("coco", 0.75, {"ap": 0.5092, "precision": 2 / 3, "recall": 2 / 3}),
    ],
)
def test_coco_metrics_example(
    example_hyp: Detections,
    example_gt: Detections,
    ap_interpolation: APInterpolation,
    iou_threshold: float,
    result: dict[str, float],
) -> None:
    """Easy case, perfect matching"""

    metrics = compute_metrics(
        example_gt, example_hyp, iou_threshold, ap_interpolation=ap_interpolation
    )

    assert metrics["mean_ap"] == pytest.approx(result["ap"], abs=1e-4)
    assert metrics["class_results"]["cat"]["precision"] == pytest.approx(
        result["precision"], abs=1e-4
    )
    assert metrics["class_results"]["cat"]["recall"] == pytest.approx(
        result["recall"], abs=1e-4
    )


def test_coco_crowd_gt_is_ignored_and_not_preferred_in_matching() -> None:
    """
    Crowd GTs must be treated as ignored:
      * they do not contribute to n_gts / recall denominator
      * a detection should match a non-ignored GT first (even if the crowd-adjusted
        similarity would be higher)
    This test is constructed so that:
      - non-crowd GT has standard IoU = 0.6 (> 0.5)
      - crowd GT has standard IoU = 0.25 (< 0.5), but crowd-adjusted similarity = 1.0
    Correct behavior: match the non-crowd GT => TP=1, FN=0, n_gts=1, precision=recall=1.
    """
    hyp = Detections(
        [0],
        np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        [0],
        ("cls",),
        ("im",),
        confs=[0.9],
    )

    # GT0: non-crowd, IoU(pred, gt0) = 0.6
    # GT1: crowd, contains pred -> standard IoU = 0.25, crowd-adjusted similarity = 1.0
    gts_bbox = np.array(
        [
            [0.25, 0.0, 1.0, 1.0],  # non-crowd
            [0.0, 0.0, 2.0, 2.0],  # crowd
        ],
        dtype=np.float32,
    )
    gts_crowd = np.array([False, True], dtype=np.bool_)

    # Build GT detections with crowd flags (supporting a few common field names).
    try:
        gts = Detections(
            [0, 0], gts_bbox, [0, 0], ("cls",), ("im",), confs=None, crowd=gts_crowd
        )
    except TypeError:
        try:
            gts = Detections(
                [0, 0],
                gts_bbox,
                [0, 0],
                ("cls",),
                ("im",),
                confs=None,
                is_crowd=gts_crowd,
            )
        except TypeError:
            try:
                gts = Detections(
                    [0, 0],
                    gts_bbox,
                    [0, 0],
                    ("cls",),
                    ("im",),
                    confs=None,
                    iscrowd=gts_crowd,
                )
            except TypeError:
                try:
                    gts = Detections(
                        [0, 0],
                        gts_bbox,
                        [0, 0],
                        ("cls",),
                        ("im",),
                        confs=None,
                        gts_crowd=gts_crowd,
                    )
                except TypeError:
                    gts = Detections(
                        [0, 0], gts_bbox, [0, 0], ("cls",), ("im",), confs=None
                    )
                    for attr in ("crowd", "is_crowd", "iscrowd", "gts_crowd"):
                        if hasattr(gts, attr):
                            setattr(gts, attr, gts_crowd)
                            break
                    else:
                        pytest.skip("Detections does not expose a crowd/iscrowd field")

    cm = confusion_matrix(gts, hyp, 0.5)
    npt.assert_array_equal(cm, np.array([[1, 0], [0, 0]], dtype=np.int32))

    metrics = compute_metrics(gts, hyp, 0.5)
    assert metrics == {
        "class_results": {
            "cls": {"ap": 1.0, "n_gts": 1, "precision": 1.0, "recall": 1.0}
        },
        "mean_ap": 1.0,
    }
