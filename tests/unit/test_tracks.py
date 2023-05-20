import tempfile
from typing import Optional

import numpy as np
import pytest

from evaldet import Tracks


@pytest.fixture(scope="function")
def empty_tracks() -> Tracks:
    return Tracks([], [], [])


@pytest.fixture(scope="function")
def tracks_with_one_item() -> Tracks:
    tracks = Tracks([0], [0], np.array([[0, 0, 1, 1]]), [1], [0.5])
    return tracks


@pytest.fixture(scope="function")
def tracks_with_one_item_no_conf() -> Tracks:
    tracks = Tracks([0], [0], np.array([[0, 0, 1, 1]]), classes=[1])
    return tracks


@pytest.fixture(scope="function")
def tracks_with_one_item_no_class() -> Tracks:
    tracks = Tracks([0], [0], np.array([[0, 0, 1, 1]]), confs=[0.5])
    return tracks


@pytest.fixture(scope="function")
def tracks_with_one_item_nothing() -> Tracks:
    tracks = Tracks([0], [0], np.array([[0, 0, 1, 1]]))
    return tracks


##############################
# Test errors
##############################


def test_mismatch_len_ids_detections() -> None:
    with pytest.raises(ValueError, match="`bboxes` and `ids` should"):
        Tracks([0, 0], [0, 1], np.array([[0, 0, 1, 1]]))


def test_mismatch_len_ids_frame_nums() -> None:
    with pytest.raises(ValueError, match="`ids` and `frame_nums` should"):
        Tracks([0], [0, 1], np.array([[0, 0, 1, 1]]))


def test_mismatch_len_ids_classes() -> None:
    with pytest.raises(ValueError, match="If `classes` is given, it should contain"):
        Tracks([0], [0], np.array([[0, 0, 1, 1]]), classes=[0, 0])


def test_mismatch_len_ids_confs() -> None:
    with pytest.raises(ValueError, match="If `confs` is given, it should contain"):
        Tracks([0], [0], np.array([[0, 0, 1, 1]]), confs=[0, 0])


def test_wrong_num_cols_detections() -> None:
    with pytest.raises(ValueError, match="Each row of `bboxes`"):
        Tracks([0], [0], np.array([[0, 0, 1]]))


def test_non_existent_frame_id(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(KeyError, match="The frame 10"):
        tracks_with_one_item[10]


def test_negative_frame_id(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(ValueError, match="Indexing with negative values is"):
        tracks_with_one_item[-1]


def test_filter_frame_wrong_shape(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(ValueError, match="Shape of the filter should equal"):
        tracks_with_one_item.filter(np.ones((100,)).astype(bool))


def test_get_slice_step_provided(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(ValueError, match="Slicing with the step"):
        tracks_with_one_item[0:1:1]


def test_get_slice_negative_start(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(ValueError, match="Slicing with negative indices"):
        tracks_with_one_item[-1:1]


def test_get_slice_negtive_stop(tracks_with_one_item: Tracks) -> None:
    with pytest.raises(ValueError, match="Slicing with negative indices"):
        tracks_with_one_item[0:-1]


##############################
# Test core functionality
##############################


def test_init_single() -> None:
    tr = Tracks(
        ids=[0],
        frame_nums=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=[0.9],
        classes=[1],
    )

    np.testing.assert_array_equal(tr.ids, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(tr.frame_nums, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(tr.bboxes, np.array([[0, 0, 1, 1]], dtype=np.float32))
    np.testing.assert_array_equal(tr.confs, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(tr.classes, np.array([1], dtype=np.int32))


def test_init_full() -> None:
    tr = Tracks(
        ids=[0, 1],
        frame_nums=[1, 0],
        bboxes=np.array([[2, 0, 1, 1], [0, 0, 1, 1]]),
        confs=[0.9, 0.99],
        classes=[1, 0],
    )

    # See that it was sorted by frame numbers
    np.testing.assert_array_equal(tr.ids, np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(tr.frame_nums, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        tr.bboxes, np.array([[0, 0, 1, 1], [2, 0, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(tr.confs, np.array([0.99, 0.9], dtype=np.float32))
    np.testing.assert_array_equal(tr.classes, np.array([0, 1], dtype=np.int32))


def test_init_empty() -> None:
    tr = Tracks(ids=[], frame_nums=[], bboxes=[])

    np.testing.assert_array_equal(tr.ids, np.zeros((0,), dtype=np.int32))
    np.testing.assert_array_equal(tr.frame_nums, np.zeros((0,), dtype=np.int32))
    np.testing.assert_array_equal(tr.bboxes, np.zeros((0, 4), dtype=np.float32))
    np.testing.assert_array_equal(tr.confs, np.zeros((0,), dtype=np.float32))
    np.testing.assert_array_equal(tr.classes, np.zeros((0,), dtype=np.int32))


def test_init_no_confs() -> None:
    tr = Tracks(
        ids=[0],
        frame_nums=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=None,
        classes=[1],
    )
    np.testing.assert_array_equal(tr.confs, np.array([1.0], dtype=np.float32))


def test_init_no_classes() -> None:
    tr = Tracks(
        ids=[0],
        frame_nums=[0],
        bboxes=np.array([[0, 0, 1, 1]]),
        confs=[0.9],
        classes=None,
    )
    np.testing.assert_array_equal(tr.classes, np.array([0], dtype=np.int32))


def test_filter() -> None:
    tr = Tracks(
        ids=[0, 1],
        frame_nums=[1, 0],
        bboxes=np.array([[2, 0, 1, 1], [0, 0, 1, 1]]),
        confs=[0.9, 0.99],
        classes=[1, 0],
    )

    tr.filter(np.array([False, True]))

    np.testing.assert_array_equal(tr.ids, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(tr.frame_nums, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(tr.bboxes, np.array([[2, 0, 1, 1]], dtype=np.float32))
    np.testing.assert_array_equal(tr.confs, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(tr.classes, np.array([1], dtype=np.int32))

    tr._frame_ind_dict == {1: (0, 1)}


def test_contains_true(tracks_with_one_item: Tracks) -> None:
    assert 0 in tracks_with_one_item


def test_contains_false(tracks_with_one_item: Tracks) -> None:
    assert 1 not in tracks_with_one_item


def test_getitem_one_item(tracks_with_one_item: Tracks) -> None:
    item = tracks_with_one_item[0]

    assert item.ids == [0]
    assert item.classes == [1]
    np.testing.assert_array_equal(item.bboxes, np.array([[0, 0, 1, 1]]))


def test_getitem_one_item_no_class(tracks_with_one_item_no_class: Tracks) -> None:
    item = tracks_with_one_item_no_class[0]

    assert item.ids == [0]
    np.testing.assert_array_equal(item.bboxes, np.array([[0, 0, 1, 1]]))
    np.testing.assert_array_equal(item.classes, np.array([0]))
    np.testing.assert_array_equal(item.confs, np.array([0.5]))


def test_getitem_one_item_no_conf(tracks_with_one_item_no_conf: Tracks) -> None:
    item = tracks_with_one_item_no_conf[0]

    assert item.ids == [0]
    np.testing.assert_array_equal(item.bboxes, np.array([[0, 0, 1, 1]]))
    np.testing.assert_array_equal(item.classes, np.array([1]))
    np.testing.assert_array_equal(item.confs, np.array([1.0]))


def test_getitem_one_item_nothing(tracks_with_one_item_nothing: Tracks) -> None:
    item = tracks_with_one_item_nothing[0]

    assert item.ids == [0]
    np.testing.assert_array_equal(item.bboxes, np.array([[0, 0, 1, 1]]))
    np.testing.assert_array_equal(item.classes, np.array([0]))
    np.testing.assert_array_equal(item.confs, np.array([1.0]))


def test_slice_normal1(sample_tracks: Tracks) -> None:
    sliced_tracks = sample_tracks[660:662]
    assert sliced_tracks.frames == {660, 661}
    assert sliced_tracks.all_classes == {2}
    np.testing.assert_almost_equal(
        sliced_tracks[661].bboxes,
        np.array(
            [
                [320.98, 105.24, 44.67, 35.71],
                [273.1, 88.88, 55.7, 24.52],
                [374.69, 80.78, 26.4, 22.23],
            ],
            dtype=np.float32,
        ),
    )


def test_slice_normal2(sample_tracks: Tracks) -> None:
    sliced_tracks = sample_tracks[660:800]
    assert sliced_tracks.frames == {660, 661}
    assert sliced_tracks.all_classes == {2}
    np.testing.assert_almost_equal(
        sliced_tracks[661].bboxes,
        np.array(
            [
                [320.98, 105.24, 44.67, 35.71],
                [273.1, 88.88, 55.7, 24.52],
                [374.69, 80.78, 26.4, 22.23],
            ],
            dtype=np.float32,
        ),
    )


def test_slice_empty_result(tracks_with_one_item: Tracks) -> None:
    sliced_tracks = tracks_with_one_item[100:111]

    assert len(sliced_tracks.classes) == 0
    assert len(sliced_tracks.confs) == 0
    assert len(sliced_tracks.ids) == 0
    assert len(sliced_tracks.bboxes) == 0
    assert len(sliced_tracks.frame_nums) == 0


def test_slice_on_empty_tracks(empty_tracks: Tracks) -> None:
    sliced_tracks = empty_tracks[0:1]

    assert len(sliced_tracks.classes) == 0
    assert len(sliced_tracks.confs) == 0
    assert len(sliced_tracks.ids) == 0
    assert len(sliced_tracks.bboxes) == 0
    assert len(sliced_tracks.frame_nums) == 0


##############################
# Test creation from files
##############################


def test_error_convert_number_mot_cvat() -> None:
    """
    Test that an error is raised when there is an error
    converting values in CVAT, because they are not numbers.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/test.csv", "w") as tmpfile:
            tmpfile.write("0, 0, 0, 0, 1, 1, 0, car")

        with pytest.raises(ValueError, match="Error when converting values"):
            Tracks.from_mot_cvat(f"{tmpdir}/test.csv")


def test_read_csv(sample_tracks: Tracks) -> None:
    tracks = Tracks.from_csv(
        "tests/data/tracks/basic_csv.csv",
        ["frame", "id", "xmin", "ymin", "width", "height", "conf", "class"],
    )

    assert tracks.frames == sample_tracks.frames

    np.testing.assert_array_equal(tracks[660].ids, sample_tracks[660].ids)
    np.testing.assert_array_equal(tracks[661].ids, sample_tracks[661].ids)
    np.testing.assert_array_equal(tracks[800].ids, sample_tracks[800].ids)

    np.testing.assert_array_almost_equal(
        tracks[660].bboxes,
        sample_tracks[660].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661].bboxes,
        sample_tracks[661].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800].bboxes,
        sample_tracks[800].bboxes,
        decimal=4,
    )

    np.testing.assert_array_equal(tracks[660].classes, sample_tracks[660].classes)
    np.testing.assert_array_equal(tracks[661].classes, sample_tracks[661].classes)
    np.testing.assert_array_equal(tracks[800].classes, sample_tracks[800].classes)

    np.testing.assert_array_equal(tracks[660].confs, sample_tracks[660].confs)
    np.testing.assert_array_equal(tracks[661].confs, sample_tracks[661].confs)
    np.testing.assert_array_equal(tracks[800].confs, sample_tracks[800].confs)

    assert tracks._frame_ind_dict == sample_tracks._frame_ind_dict


def test_read_mot_cvat(sample_tracks: Tracks) -> None:
    tracks = Tracks.from_mot_cvat("tests/data/tracks/cvat_mot_sample.csv")

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660].ids) == set(sample_tracks[660].ids)
    assert set(tracks[661].ids) == set(sample_tracks[661].ids)
    assert set(tracks[800].ids) == set(sample_tracks[800].ids)

    np.testing.assert_array_almost_equal(
        tracks[660].bboxes,
        sample_tracks[660].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661].bboxes,
        sample_tracks[661].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800].bboxes,
        sample_tracks[800].bboxes,
        decimal=4,
    )

    assert set(tracks[660].classes) == set(sample_tracks[660].classes)
    assert set(tracks[661].classes) == set(sample_tracks[661].classes)
    assert set(tracks[800].classes) == set(sample_tracks[800].classes)

    assert tracks._frame_ind_dict == sample_tracks._frame_ind_dict


def test_read_mot(sample_tracks: Tracks) -> None:
    tracks = Tracks.from_mot("tests/data/tracks/mot_sample.csv")

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660].ids) == set(sample_tracks[660].ids)
    assert set(tracks[661].ids) == set(sample_tracks[661].ids)
    assert set(tracks[800].ids) == set(sample_tracks[800].ids)

    np.testing.assert_array_almost_equal(
        tracks[660].bboxes,
        sample_tracks[660].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661].bboxes,
        sample_tracks[661].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800].bboxes,
        sample_tracks[800].bboxes,
        decimal=4,
    )

    assert tracks._frame_ind_dict == sample_tracks._frame_ind_dict


def test_error_ua_detrac_no_class_list() -> None:
    with pytest.raises(ValueError, match="If you provide `classes_attr_name`,"):
        Tracks.from_ua_detrac(
            "tests/data/tracks/ua_detrac_sample.xml",
            classes_attr_name="vehicle_type",
        )


def test_read_ua_detrac(sample_tracks: Tracks) -> None:
    tracks = Tracks.from_ua_detrac(
        "tests/data/tracks/ua_detrac_sample.xml",
        classes_attr_name="vehicle_type",
        classes_list=["Taxi", "Bike", "Car"],
    )
    tracks_no_cls = Tracks.from_ua_detrac("tests/data/tracks/ua_detrac_sample.xml")

    assert tracks.frames == sample_tracks.frames
    assert tracks_no_cls.frames == sample_tracks.frames

    assert set(tracks[660].ids) == set(sample_tracks[660].ids)
    assert set(tracks[661].ids) == set(sample_tracks[661].ids)
    assert set(tracks[800].ids) == set(sample_tracks[800].ids)

    assert set(tracks_no_cls[660].ids) == set(sample_tracks[660].ids)
    assert set(tracks_no_cls[661].ids) == set(sample_tracks[661].ids)
    assert set(tracks_no_cls[800].ids) == set(sample_tracks[800].ids)

    np.testing.assert_array_almost_equal(
        tracks[660].bboxes,
        sample_tracks[660].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661].bboxes,
        sample_tracks[661].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800].bboxes,
        sample_tracks[800].bboxes,
        decimal=4,
    )

    assert set(tracks[660].classes) == set(sample_tracks[660].classes)
    assert set(tracks[661].classes) == set(sample_tracks[661].classes)
    assert set(tracks[800].classes) == set(sample_tracks[800].classes)

    assert tracks._frame_ind_dict == sample_tracks._frame_ind_dict


def test_read_cvat_video(sample_tracks: Tracks) -> None:
    tracks = Tracks.from_cvat_video(
        "tests/data/tracks/cvat_video_sample.xml", classes_list=["Taxi", "Bike", "Car"]
    )

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660].ids) == set(sample_tracks[660].ids)
    assert set(tracks[661].ids) == set(sample_tracks[661].ids)
    assert set(tracks[800].ids) == set(sample_tracks[800].ids)

    np.testing.assert_array_almost_equal(
        tracks[660].bboxes,
        sample_tracks[660].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661].bboxes,
        sample_tracks[661].bboxes,
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800].bboxes,
        sample_tracks[800].bboxes,
        decimal=4,
    )

    assert set(tracks[660].classes) == set(sample_tracks[660].classes)
    assert set(tracks[661].classes) == set(sample_tracks[661].classes)
    assert set(tracks[800].classes) == set(sample_tracks[800].classes)

    assert tracks._frame_ind_dict == sample_tracks._frame_ind_dict
