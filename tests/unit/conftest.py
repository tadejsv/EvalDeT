import numpy as np
import pytest

from evaldet.detections import Detections
from evaldet.tracks import Tracks


@pytest.fixture
def missing_frame_pair() -> tuple[Tracks, Tracks]:
    tracks_full = Tracks(
        ids=[0, 0], frame_nums=[0, 1], bboxes=np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    )
    tracks_missing = Tracks(ids=[0], frame_nums=[0], bboxes=np.array([[0, 0, 1, 1]]))

    return tracks_full, tracks_missing


@pytest.fixture
def simple_case() -> tuple[Tracks, Tracks]:
    gt = Tracks(
        ids=[0, 1, 0, 1, 0, 1],
        frame_nums=[0, 0, 1, 1, 2, 2],
        bboxes=np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
            ]
        ),
    )
    hyp = Tracks(
        ids=[0, 1, 0, 1, 2, 1],
        frame_nums=[0, 0, 1, 1, 2, 2],
        bboxes=np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [0.1, 0.1, 1, 1],
                [1, 1, 1, 1],
                [0.05, 0.05, 1, 1],
                [2, 2, 1, 1],
            ]
        ),
    )
    return gt, hyp


@pytest.fixture(scope="module")
def sample_tracks() -> Tracks:
    tracks = Tracks(
        frame_nums=[660] * 3 + [661] * 3 + [800] * 2,
        ids=[1, 2, 3, 1, 2, 3, 2, 4],
        bboxes=np.array(
            [
                [323.83, 104.06, 43.77, 35.43],
                [273.1, 88.77, 55.59, 24.32],
                [375.24, 80.43, 26.41, 22.24],
                [320.98, 105.24, 44.67, 35.71],
                [273.1, 88.88, 55.7, 24.52],
                [374.69, 80.78, 26.4, 22.23],
                [329.27, 96.65, 56.53, 32.45],
                [0.0, 356.7, 76.6, 122.67],
            ]
        ),
        classes=[2] * 8,
        confs=[0.9] * 8,
    )
    return tracks


@pytest.fixture(scope="function")
def sample_detections() -> Detections:
    dets = Detections(
        image_ids=[0] * 3 + [1] * 3 + [2] * 2,
        bboxes=np.array(
            [
                [323.83, 104.06, 43.77, 35.43],
                [273.1, 88.77, 55.59, 24.32],
                [375.24, 80.43, 26.41, 22.24],
                [320.98, 105.24, 44.67, 35.71],
                [273.1, 88.88, 55.7, 24.52],
                [374.69, 80.78, 26.4, 22.23],
                [329.27, 96.65, 56.53, 32.45],
                [0.0, 356.7, 76.6, 122.67],
            ]
        ),
        classes=[2] * 8,
        confs=[0.9] * 8,
        class_names=["horse", "cat", "car"],
        image_names=["car1", "car2", "car3"],
    )
    return dets
