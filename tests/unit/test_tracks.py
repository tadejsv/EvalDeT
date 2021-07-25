import numpy as np
import pytest

from evaldet import Tracks


@pytest.fixture(scope="function")
def empty_tracks():
    return Tracks()


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


def test_add_one_observation(empty_tracks: Tracks):
    pass


def test_add_more_observations(empty_tracks: Tracks):
    pass
