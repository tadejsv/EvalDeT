import numpy as np
import numpy.testing as npt
import pytest
from deepdiff import DeepDiff

from evaldet.detections import Detections
from evaldet.det.coco import COCOMetrics


@pytest.fixture(scope="function")
def empty_dets() -> Detections:
    return Detections([], [], [], confs=[], class_names=("cls",), image_names=tuple())


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def example_hyp() -> Detections:
    pass


@pytest.fixture(scope="function")
def example_gt() -> Detections:
    pass


#########################################
# Test Errors
#########################################


def test_coco_different_class_names_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("slc",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=None)

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        coco.confusion_matrix(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        coco.compute_metrics(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        coco.compute_coco_summary(gts, hyp)


def test_coco_no_hyp_conf_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=None)
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=None)

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`confs` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`confs` must be provided"):
        coco.compute_metrics(gts, hyp, 0.5)

    with pytest.raises(ValueError, match="`confs` must be provided"):
        coco.compute_coco_summary(gts, hyp)


#########################################
# Test Basic Funcionality
#########################################


def test_coco_cm_empty_gts_hyp(empty_dets: Detections) -> None:
    coco = COCOMetrics()
    cm = coco.confusion_matrix(empty_dets, empty_dets, 0.5)

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

    coco = COCOMetrics()
    cm = coco.confusion_matrix(empty_dets, hyp, 0.5)

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

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, empty_dets, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_empty_image() -> None:
    gts = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=None
    )
    hyp = Detections(
        [0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im", "im2"), confs=[0.1]
    )

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[1, 0], [0, 0]], dtype=np.int32))


def test_coco_metrics_empty_gts_hyp(empty_dets: Detections) -> None:
    coco = COCOMetrics()
    metrics = coco.compute_metrics(empty_dets, empty_dets, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
        "mean_ap": None,
    }


def test_coco_metrics_empty_gts(empty_dets: Detections) -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    coco = COCOMetrics()
    metrics = coco.compute_metrics(empty_dets, hyp, 0.5)

    assert metrics == {
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
        "mean_ap": None,
    }


def test_coco_metrics_empty_preds(empty_dets: Detections) -> None:
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, empty_dets, 0.5)

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

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 0.5)

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

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_hyp() -> None:
    hyp = Detections(
        [0], [np.array([1, 1, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_matching() -> None:
    """Hyp is matched to an ingored gt - by COCO rules this means hyp is ignored too"""
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 1e-9, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [0, 0]], dtype=np.int32))


def test_coco_cm_no_matching() -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 1], [1, 0]], dtype=np.int32))


def test_coco_metrics_all_ignored_gts() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections(
        [0], [np.array([0, 0, 0.1, 0.1])], [0], ("cls",), ("im",), confs=[0.1]
    )

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 0.5, (1, 100))

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

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 0.5, (1, 100))

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

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 1e-9, (1, 100))

    assert metrics == {
        "mean_ap": None,
        "class_results": {
            "cls": {"ap": None, "n_gts": 0, "precision": None, "recall": None}
        },
    }


def test_coco_metrics_no_matching() -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])
    gts = Detections([0], [np.array([1, 1, 1, 1])], [0], ("cls",), ("im",), confs=[0.1])

    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 0.5)

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
    coco = COCOMetrics()
    metrics = coco.compute_metrics(gts, hyp, 0.5)

    assert metrics == {
        "mean_ap": 1.0,
        "class_results": {
            "cls1": {"ap": 1.0, "n_gts": 1, "precision": 1.0, "recall": 1.0},
            "cls2": {"ap": None, "n_gts": 0, "precision": None, "recall": None},
        },
    }


def test_coco_cm_normal1(normal_hyp_1: Detections, normal_gt_1: Detections) -> None:
    """Easy case, perfect matching"""

    coco = COCOMetrics()
    cm = coco.confusion_matrix(normal_gt_1, normal_hyp_1, 0.5)

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
    coco = COCOMetrics()
    cm = coco.confusion_matrix(normal_gt_2, normal_hyp_2, 0.5, (1, 100))

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
    "ap_interpolation,result",
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
    ap_interpolation: str,
    result: dict,
) -> None:
    coco = COCOMetrics(ap_interpolation)
    metrics = coco.compute_metrics(normal_gt_1, normal_hyp_1, 0.5)

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    "ap_interpolation,result",
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
    ap_interpolation: str,
    result: dict,
) -> None:
    coco = COCOMetrics(ap_interpolation)
    metrics = coco.compute_metrics(
        normal_gt_2, normal_hyp_2, 0.5, area_range=(0.5, 100)
    )

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    "ap_interpolation,result",
    [
        (
            "pascal",
            {
                "mean_ap": 13 / 54,
                "mean_ap_per_class": {
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
                "mean_ap_per_class": {
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
    ap_interpolation: str,
    result: dict,
) -> None:
    coco = COCOMetrics(ap_interpolation)
    metrics = coco.compute_coco_summary(normal_gt_1, normal_hyp_1)

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )


@pytest.mark.parametrize(
    "ap_interpolation,result",
    [
        (
            "pascal",
            {
                "mean_ap": 7 / 72,
                "mean_ap_per_class": {
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
                "mean_ap_per_class": {
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
    ap_interpolation: str,
    result: dict,
) -> None:
    coco = COCOMetrics(ap_interpolation)
    metrics = coco.compute_coco_summary(normal_gt_2, normal_hyp_2)

    assert not DeepDiff(
        metrics,
        result,
        ignore_order=True,
        significant_digits=6,
        ignore_numeric_type_changes=True,
    )
