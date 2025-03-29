import numpy as np
import pytest

from evaldet import Detections


@pytest.fixture
def empty_dets() -> Detections:
    return Detections([], [], [], class_names=(), image_names=())


@pytest.fixture
def dets_with_one_item() -> Detections:
    return Detections(
        [0],
        np.array([[0, 0, 1, 1]]),
        [1],
        confs=[0.5],
        class_names=("cls1", "cls2"),
        image_names=("im"),
    )


@pytest.fixture
def dets_with_one_item_no_conf() -> Detections:
    return Detections(
        [0],
        np.array([[0, 0, 1, 1]]),
        classes=[1],
        class_names=("cls1", "cls2"),
        image_names=("im"),
    )


##############################
# Test errors
##############################


def test_mismatch_len_imgs_bboxes() -> None:
    with pytest.raises(ValueError, match="`image_ids` and `bboxes`"):
        Detections(
            [0, 0],
            np.array([[0, 0, 1, 1]]),
            classes=[0, 1],
            confs=[0, 0],
            class_names=("cls",),
            image_names=("im",),
        )


def test_mismatch_len_ids_classes() -> None:
    with pytest.raises(ValueError, match="`image_ids` and `classes`"):
        Detections(
            [0],
            np.array([[0, 0, 1, 1]]),
            classes=[0, 0],
            confs=[0],
            class_names=("cls",),
            image_names=("im",),
        )


def test_mismatch_len_ids_confs() -> None:
    with pytest.raises(ValueError, match="If `confs` is given, it should contain"):
        Detections(
            [0],
            np.array([[0, 0, 1, 1]]),
            classes=[0],
            confs=[0, 0],
            class_names=("cls",),
            image_names=("im",),
        )


def test_wrong_num_cols_detections() -> None:
    with pytest.raises(ValueError, match="Each row of `bboxes`"):
        Detections(
            [0],
            np.array([[0, 0, 1]]),
            classes=[0],
            confs=[0],
            class_names=("cls",),
            image_names=("im",),
        )


def test_filter_frame_wrong_shape(dets_with_one_item: Detections) -> None:
    with pytest.raises(ValueError, match="Shape of the filter should equal"):
        dets_with_one_item.filter(np.ones((100,)).astype(bool))


def test_not_enough_image_names1() -> None:
    with pytest.raises(ValueError, match="The number of image names"):
        Detections(
            [0, 1],
            np.array([[0, 0, 1, 1], [1, 0, 1, 1]]),
            classes=[0, 1],
            image_names=["im"],
            class_names=["cls1", "cls2"],
        )


def test_not_enough_image_names2() -> None:
    with pytest.raises(ValueError, match="The number of image names"):
        Detections(
            [0, 2],
            np.array([[0, 0, 1, 1], [1, 0, 1, 1]]),
            classes=[0, 1],
            image_names=["im1", "im2"],
            class_names=["cls1", "cls2"],
        )


def test_not_enough_class_names1() -> None:
    with pytest.raises(ValueError, match="The number of class names"):
        Detections(
            [0, 1],
            np.array([[0, 0, 1, 1], [1, 0, 1, 1]]),
            classes=[0, 1],
            image_names=["im1", "im2"],
            class_names=["c1"],
        )


def test_not_enough_class_names2() -> None:
    with pytest.raises(ValueError, match="The number of class names"):
        Detections(
            [0, 1],
            np.array([[0, 0, 1, 1], [1, 0, 1, 1]]),
            classes=[0, 2],
            image_names=["im1", "im2"],
            class_names=["c1", "c2"],
        )


##############################
# Test core functionality
##############################


def test_init_single() -> None:
    det = Detections(
        image_ids=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=[0.9],
        classes=[1],
        image_names=("im1",),
        class_names=("cls1", "cls2"),
    )

    assert det.confs is not None
    np.testing.assert_array_equal(det.image_ids, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(
        det.bboxes, np.array([[0, 0, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(det.confs, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(det.classes, np.array([1], dtype=np.int32))

    assert det.num_classes == 1
    assert det.num_dets == 1
    assert det.num_images == 1

    assert det.image_ind_dict == {0: (0, 1)}

    assert det.class_names == ("cls1", "cls2")
    assert det.image_names == ("im1",)


def test_init_full() -> None:
    det = Detections(
        image_ids=[1, 0],
        bboxes=np.array([[2, 0, 1, 1], [0, 0, 1, 1]]),
        confs=[0.9, 0.99],
        classes=[1, 0],
        image_names=("im1", "im2"),
        class_names=("cls1", "cls2"),
    )

    assert det.confs is not None

    # See that it was sorted by frame numbers
    np.testing.assert_array_equal(det.image_ids, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        det.bboxes, np.array([[0, 0, 1, 1], [2, 0, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(det.confs, np.array([0.99, 0.9], dtype=np.float32))
    np.testing.assert_array_equal(det.classes, np.array([0, 1], dtype=np.int32))

    assert det.num_classes == 2
    assert det.num_dets == 2
    assert det.num_images == 2

    assert det.image_ind_dict == {0: (0, 1), 1: (1, 2)}

    assert det.class_names == ("cls1", "cls2")
    assert det.image_names == ("im1", "im2")


def test_init_empty() -> None:
    det = Detections(
        image_ids=[], classes=[], bboxes=[], class_names=["cls"], image_names=["img"]
    )

    assert det.confs is not None

    np.testing.assert_array_equal(det.image_ids, np.zeros((0,), dtype=np.int32))
    np.testing.assert_array_equal(det.bboxes, np.zeros((0, 4), dtype=np.float32))
    np.testing.assert_array_equal(det.confs, np.zeros((0,), dtype=np.float32))
    np.testing.assert_array_equal(det.classes, np.zeros((0,), dtype=np.int32))

    assert det.class_names == ("cls",)
    assert det.image_names == ("img",)


def test_init_empty_no_names() -> None:
    det = Detections(
        image_ids=[], classes=[], bboxes=[], class_names=[], image_names=[]
    )

    assert det.confs is not None

    np.testing.assert_array_equal(det.image_ids, np.zeros((0,), dtype=np.int32))
    np.testing.assert_array_equal(det.bboxes, np.zeros((0, 4), dtype=np.float32))
    np.testing.assert_array_equal(det.confs, np.zeros((0,), dtype=np.float32))
    np.testing.assert_array_equal(det.classes, np.zeros((0,), dtype=np.int32))

    assert det.class_names == ()
    assert det.image_names == ()


def test_init_no_confs() -> None:
    det = Detections(
        image_ids=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=None,
        classes=[1],
        image_names=("im1", "im2"),
        class_names=("cls1", "cls2"),
    )
    assert det.confs is None


def test_init_full_extra_names() -> None:
    det = Detections(
        image_ids=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=[0.9],
        classes=[1],
        class_names=["cl1", "cl2", "cl3"],
        image_names=["im1", "im2"],
    )

    assert det.class_names == ("cl1", "cl2", "cl3")
    assert det.image_names == ("im1", "im2")


def test_filter() -> None:
    det = Detections(
        image_ids=[1, 0],
        bboxes=np.array([[2, 0, 1, 1], [0, 0, 1, 1]]),
        confs=[0.9, 0.99],
        classes=[1, 0],
        image_names=["one", "two"],
        class_names=["c1", "c2"],
    )

    detf = det.filter(np.array([False, True]))

    assert detf.confs is not None

    np.testing.assert_array_equal(detf.image_ids, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(
        detf.bboxes, np.array([[2, 0, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(detf.confs, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(detf.classes, np.array([1], dtype=np.int32))

    assert detf.image_ind_dict == {1: (0, 1)}

    assert detf.image_names == det.image_names
    assert detf.class_names == det.class_names


def test_filter_no_confs() -> None:
    det = Detections(
        image_ids=[1, 0],
        bboxes=np.array([[2, 0, 1, 1], [0, 0, 1, 1]]),
        classes=[1, 0],
        image_names=["one", "two"],
        class_names=["c1", "c2"],
    )

    detf = det.filter(np.array([False, True]))

    np.testing.assert_array_equal(detf.image_ids, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(
        detf.bboxes, np.array([[2, 0, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(detf.classes, np.array([1], dtype=np.int32))

    assert detf.confs is None
