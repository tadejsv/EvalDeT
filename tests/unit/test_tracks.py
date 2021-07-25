import numpy as np
import pytest

from evaldet import Tracks


@pytest.fixture(scope="function")
def empty_tracks():
    return Tracks()


@pytest.fixture(scope="function")
def tracks_with_one_item():
    tracks = Tracks()
    tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]), [1])
    return tracks


@pytest.fixture(scope="function")
def tracks_with_one_item_no_class():
    tracks = Tracks()
    tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]))
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
    with pytest.raises(ValueError, match="The frame 10"):
        tracks_with_one_item[10]


##############################
# Test core functionality
##############################


def test_add_one_observation(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]), [1])

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set([1])
    assert empty_tracks._last_frame == 0


def test_add_one_observation_no_class(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0], np.array([[0, 0, 1, 1]]))

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set()
    assert empty_tracks._last_frame == 0


def test_add_more_observations(empty_tracks: Tracks):
    empty_tracks.add_frame(0, [0, 1], np.array([[0, 0, 1, 1], [0, 0, 1, 1]]), [1, 1])

    assert len(empty_tracks) == 1
    assert empty_tracks.frames == [0]
    assert empty_tracks.all_classes == set([1])
    assert empty_tracks._last_frame == 0


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


def test_add_second_observation_no_class(tracks_with_one_item: Tracks):
    tracks_with_one_item.add_frame(2, [2], np.array([[0, 0, 1, 1]]))

    assert len(tracks_with_one_item) == 2
    assert tracks_with_one_item.frames == [0, 2]
    assert tracks_with_one_item.all_classes == set([1])
    assert tracks_with_one_item._last_frame == 2


##############################
# Test creation from files
##############################
