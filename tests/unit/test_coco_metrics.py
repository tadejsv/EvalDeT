import numpy as np
import numpy.testing as npt
import pytest

from evaldet.detections import Detections
from evaldet.det.coco import COCOMetrics


@pytest.fixture(scope="function")
def empty_dets() -> Detections:
    return Detections([], [], [], [], ("cls",), tuple())


#########################################
# Test Errors
#########################################


def test_coco_cm_no_image_names_gts_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, ("cls",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`image_names` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)


def test_coco_cm_no_image_names_hyp_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("cls",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, ("cls",), ("im",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`image_names` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)


def test_coco_cm_no_class_names_gts_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, None, ("im",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`class_names` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)


def test_coco_cm_no_class_names_hyp_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], None, ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, ("cls",), ("im",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`class_names` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)


def test_coco_cm_different_class_names_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("slc",), ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, ("cls",), ("im",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`class_names` must be the same"):
        coco.confusion_matrix(gts, hyp, 0.5)


def test_coco_cm_no_hyp_conf_error() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], None, ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], None, ("cls",), ("im",))

    coco = COCOMetrics()

    with pytest.raises(ValueError, match="`confs` must be provided"):
        coco.confusion_matrix(gts, hyp, 0.5)


#########################################
# Test Basic Funcionality
#########################################


def test_coco_cm_empty_gts_hyp(empty_dets: Detections) -> None:
    coco = COCOMetrics()
    cm = coco.confusion_matrix(empty_dets, empty_dets, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 0], [0, 0]], dtype=np.int32))


def test_coco_cm_empty_gts(empty_dets: Detections) -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(empty_dets, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_coco_cm_empty_preds(empty_dets: Detections) -> None:
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, empty_dets, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_gts() -> None:
    hyp = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 0.1, 0.1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 1], [0, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_hyp() -> None:
    hyp = Detections([0], [np.array([1, 1, 0.1, 0.1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 1, 1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [1, 0]], dtype=np.int32))


def test_coco_cm_all_ignored_matching() -> None:
    """Hyp is matched to an ingored gt - by COCO rules this means hyp is ignored too"""
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([0, 0, 0.1, 0.1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 1e-9, (1, 100))

    npt.assert_array_equal(cm, np.array([[0, 0], [0, 0]], dtype=np.int32))


def test_coco_cm_no_matching() -> None:
    hyp = Detections([0], [np.array([0, 0, 1, 1])], [0], [0.1], ("cls",), ("im",))
    gts = Detections([0], [np.array([1, 1, 1, 1])], [0], [0.1], ("cls",), ("im",))

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5)

    npt.assert_array_equal(cm, np.array([[0, 1], [1, 0]], dtype=np.int32))


def test_coco_cm_normal1() -> None:
    """Easy case, perfect matching"""
    hyp = Detections(
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
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ("cls1", "cls2", "cls3"),
        ("im1", "im2", "im0"),
    )
    gts = Detections(
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
        None,
        ("cls1", "cls2", "cls3"),
        ("im0", "im1", "im2"),
    )

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5)

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


def test_coco_cm_normal2() -> None:
    """Easy case, some not matching, some ignored"""
    hyp = Detections(
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
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ("cls1", "cls2", "cls3"),
        ("im1", "im2", "im0", "im4"),
    )
    gts = Detections(
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
        None,
        ("cls1", "cls2", "cls3"),
        ("im0", "im1", "im2", "im5"),
    )

    coco = COCOMetrics()
    cm = coco.confusion_matrix(gts, hyp, 0.5, (1, 100))

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
