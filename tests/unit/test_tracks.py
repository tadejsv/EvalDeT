import tempfile

import numpy as np
import pytest

from evaldet import Tracks


@pytest.fixture(scope="function")
def empty_tracks() -> Tracks:
    return Tracks()


@pytest.fixture(scope="function")
def tracks_with_one_item() -> Tracks:
    tracks = Tracks()
    tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]), [1])
    return tracks


@pytest.fixture(scope="function")
def tracks_with_one_item_no_class() -> Tracks:
    tracks = Tracks()
    tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
    return tracks


@pytest.fixture(scope="module")
def sample_tracks() -> Tracks:
    tracks = Tracks()
    tracks.add_frame(
        660,
        ids=[1, 2, 3],
        detections=np.array(
            [
                [323.83, 104.06, 367.6, 139.49],
                [273.1, 88.77, 328.69, 113.09],
                [375.24, 80.43, 401.65, 102.67],
            ]
        ),
        classes=[2, 2, 2],
    )
    tracks.add_frame(
        661,
        ids=[1, 2, 3],
        detections=np.array(
            [
                [320.98, 105.24, 365.65, 140.95],
                [273.1, 88.88, 328.8, 113.4],
                [374.69, 80.78, 401.09, 103.01],
            ]
        ),
        classes=[2, 2, 2],
    )
    tracks.add_frame(
        800,
        ids=[2, 4],
        detections=np.array(
            [
                [329.27, 96.65, 385.8, 129.1],
                [0.0, 356.7, 76.6, 479.37],
            ]
        ),
        classes=[2, 2],
    )

    return tracks


##############################
# Test errors
##############################


def test_too_small_frame_num(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="You attempted to add frame -1"):
        empty_tracks.add_frame(-1, [0], np.array([[0, 0, 1, 1]]))


def test_empty_ids(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="You must pass a non-empty"):
        empty_tracks.add_frame(0, [], np.array([[0, 0, 1, 1]]))


def test_wrong_dim_detections(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="The `detections` must be a 2d"):
        empty_tracks.add_frame(0, [0, 1], np.array([0, 0, 1, 1]))


def test_mismatch_len_ids_detections(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="The `detections` and `ids` should"):
        empty_tracks.add_frame(0, [0, 1], np.array([[0, 0, 1, 1]]))


def test_mismatch_len_ids_classes(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="If `classes` is given, it should contain"):
        empty_tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]), [])


def test_wrong_num_cols_detections(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="The `detections` should be an Nx4"):
        empty_tracks.add_frame(0, [0], np.array([[0, 0, 1]]))


def test_wrong_x_coords(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="Detections have to be in the .* one of xmax"):
        empty_tracks.add_frame(0, [0], np.array([[1, 0, 0, 1]]))


def test_wrong_y_coords(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="Detections have to be in the .* one of ymax"):
        empty_tracks.add_frame(0, [0], np.array([[0, 1, 1, 0]]))


def test_non_unique_ids(empty_tracks: Tracks):
    with pytest.raises(ValueError, match="The `ids` must be unique"):
        empty_tracks.add_frame(0, [0, 0], np.array([[0, 0, 1, 1], [0, 0, 1, 1]]))


def test_non_existent_frame_id(tracks_with_one_item: Tracks):
    with pytest.raises(KeyError, match="The frame 10"):
        tracks_with_one_item[10]


##############################
# Test core functionality
##############################


def test_add_one_observation(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]), [1])

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set([1])
    assert empty_tracks.all_ids == set([0])
    assert empty_tracks.ids_count == {0: 1}
    assert empty_tracks._last_frame == 0


def test_add_one_observation_no_class(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]))

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set()
    assert empty_tracks.all_ids == set([0])
    assert empty_tracks.ids_count == {0: 1}
    assert empty_tracks._last_frame == 0


def test_add_more_observations(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [0, 0, 1, 1]]), [1, 1])

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set([1])
    assert empty_tracks.all_ids == set([0, 1])
    assert empty_tracks.ids_count == {0: 1, 1: 1}
    assert empty_tracks._last_frame == 0


def test_contains_true(tracks_with_one_item: Tracks):
    assert 0 in tracks_with_one_item


def test_contains_false(tracks_with_one_item: Tracks):
    assert 1 not in tracks_with_one_item


def test_getitem_one_item(tracks_with_one_item: Tracks):
    item = tracks_with_one_item[0]

    assert item["ids"] == [0]
    assert item["classes"] == [1]
    np.testing.assert_array_equal(item["detections"], np.array([[0, 0, 1, 1]]))


def test_getitem_one_item_no_class(tracks_with_one_item_no_class: Tracks):
    item = tracks_with_one_item_no_class[0]

    assert item["ids"] == [0]
    np.testing.assert_array_equal(item["detections"], np.array([[0, 0, 1, 1]]))
    assert "classes" not in item


def test_add_second_observation(tracks_with_one_item: Tracks):
    tracks_with_one_item.add_frame(2, [2], np.array([[0, 0, 1, 1]]), [3])

    assert len(tracks_with_one_item) == 2
    assert tracks_with_one_item.frames == [0, 2]
    assert tracks_with_one_item.all_classes == set([1, 3])
    assert tracks_with_one_item._last_frame == 2
    assert tracks_with_one_item.all_ids == set([0, 2])
    assert tracks_with_one_item.ids_count == {0: 1, 2: 1}


def test_add_second_observation_no_class(tracks_with_one_item: Tracks):
    tracks_with_one_item.add_frame(2, [2], np.array([[0, 0, 1, 1]]))

    assert len(tracks_with_one_item) == 2
    assert tracks_with_one_item.frames == [0, 2]
    assert tracks_with_one_item.all_classes == set([1])
    assert tracks_with_one_item._last_frame == 2
    assert tracks_with_one_item.all_ids == set([0, 2])
    assert tracks_with_one_item.ids_count == {0: 1, 2: 1}


##############################
# Test creation from files
##############################


def test_error_convert_number_mot_cvat():
    """
    Test that an error is raised when there is an error
    converting values in CVAT, because they are not numbers.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/test.csv", "w") as tmpfile:
            tmpfile.write("0, 0, 0, 0, 1, 1, 0, car")

        with pytest.raises(ValueError, match="Error when converting values"):
            Tracks.from_mot_cvat(f"{tmpdir}/test.csv")


def test_read_mot_cvat(sample_tracks):
    tracks = Tracks.from_mot_cvat("tests/data/tracks/cvat_mot_sample.csv")

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660]["ids"]) == set(sample_tracks[660]["ids"])
    assert set(tracks[661]["ids"]) == set(sample_tracks[661]["ids"])
    assert set(tracks[800]["ids"]) == set(sample_tracks[800]["ids"])

    np.testing.assert_array_almost_equal(
        tracks[660]["detections"],
        sample_tracks[660]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661]["detections"],
        sample_tracks[661]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800]["detections"],
        sample_tracks[800]["detections"],
        decimal=4,
    )

    assert set(tracks[660]["classes"]) == set(sample_tracks[660]["classes"])
    assert set(tracks[660]["classes"]) == set(sample_tracks[661]["classes"])
    assert set(tracks[800]["classes"]) == set(sample_tracks[800]["classes"])


def test_read_mot(sample_tracks):
    tracks = Tracks.from_mot("tests/data/tracks/mot_sample.csv")

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660]["ids"]) == set(sample_tracks[660]["ids"])
    assert set(tracks[661]["ids"]) == set(sample_tracks[661]["ids"])
    assert set(tracks[800]["ids"]) == set(sample_tracks[800]["ids"])

    np.testing.assert_array_almost_equal(
        tracks[660]["detections"],
        sample_tracks[660]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661]["detections"],
        sample_tracks[661]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800]["detections"],
        sample_tracks[800]["detections"],
        decimal=4,
    )


def test_error_ua_detrac_no_class_list():
    with pytest.raises(ValueError, match="If you provide `classes_attr_name`,"):
        Tracks.from_ua_detrac(
            "tests/data/tracks/ua_detrac_sample.xml",
            classes_attr_name="vehicle_type",
        )


def test_read_ua_detrac(sample_tracks):
    tracks = Tracks.from_ua_detrac(
        "tests/data/tracks/ua_detrac_sample.xml",
        classes_attr_name="vehicle_type",
        classes_list=["Taxi", "Bike", "Car"],
    )
    tracks_no_cls = Tracks.from_ua_detrac("tests/data/tracks/ua_detrac_sample.xml")

    assert tracks.frames == sample_tracks.frames
    assert tracks_no_cls.frames == sample_tracks.frames

    assert set(tracks[660]["ids"]) == set(sample_tracks[660]["ids"])
    assert set(tracks[661]["ids"]) == set(sample_tracks[661]["ids"])
    assert set(tracks[800]["ids"]) == set(sample_tracks[800]["ids"])

    assert set(tracks_no_cls[660]["ids"]) == set(sample_tracks[660]["ids"])
    assert set(tracks_no_cls[661]["ids"]) == set(sample_tracks[661]["ids"])
    assert set(tracks_no_cls[800]["ids"]) == set(sample_tracks[800]["ids"])

    np.testing.assert_array_almost_equal(
        tracks[660]["detections"],
        sample_tracks[660]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661]["detections"],
        sample_tracks[661]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800]["detections"],
        sample_tracks[800]["detections"],
        decimal=4,
    )

    assert set(tracks[660]["classes"]) == set(sample_tracks[660]["classes"])
    assert set(tracks[660]["classes"]) == set(sample_tracks[661]["classes"])
    assert set(tracks[800]["classes"]) == set(sample_tracks[800]["classes"])


def test_read_cvat_video(sample_tracks):
    tracks = Tracks.from_cvat_video(
        "tests/data/tracks/cvat_video_sample.xml", classes_list=["Taxi", "Bike", "Car"]
    )

    assert tracks.frames == sample_tracks.frames

    assert set(tracks[660]["ids"]) == set(sample_tracks[660]["ids"])
    assert set(tracks[661]["ids"]) == set(sample_tracks[661]["ids"])
    assert set(tracks[800]["ids"]) == set(sample_tracks[800]["ids"])

    np.testing.assert_array_almost_equal(
        tracks[660]["detections"],
        sample_tracks[660]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[661]["detections"],
        sample_tracks[661]["detections"],
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        tracks[800]["detections"],
        sample_tracks[800]["detections"],
        decimal=4,
    )

    assert set(tracks[660]["classes"]) == set(sample_tracks[660]["classes"])
    assert set(tracks[660]["classes"]) == set(sample_tracks[661]["classes"])
    assert set(tracks[800]["classes"]) == set(sample_tracks[800]["classes"])
