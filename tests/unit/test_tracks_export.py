import datetime as dt
import pathlib

import freezegun
import numpy as np
import pytest

from evaldet import Tracks


@pytest.fixture(scope="module")
def sample_tracks() -> Tracks:
    tracks = Tracks()
    tracks.add_frame(
        660,
        ids=[1, 2, 3],
        detections=np.array(
            [
                [323.83, 104.06, 43.77, 35.43],
                [273.1, 88.77, 55.59, 24.32],
                [375.24, 80.43, 26.41, 22.24],
            ]
        ),
        classes=[2, 2, 2],
        confs=[0.9, 0.9, 0.9],
    )
    tracks.add_frame(
        661,
        ids=[1, 2, 3],
        detections=np.array(
            [
                [320.98, 105.24, 44.67, 35.71],
                [273.1, 88.88, 55.7, 24.52],
                [374.69, 80.78, 26.4, 22.23],
            ]
        ),
        classes=[2, 2, 2],
        confs=[0.9, 0.9, 0.9],
    )
    tracks.add_frame(
        800,
        ids=[2, 4],
        detections=np.array(
            [[329.27, 96.65, 56.53, 32.45], [0.0, 356.7, 76.6, 122.67]]
        ),
        classes=[2, 2],
        confs=[0.9, 0.9],
    )

    return tracks


def test_export_normal_csv(
    sample_tracks: Tracks, tmp_path: pathlib.Path, data_dir: pathlib.Path
):
    sample_tracks.to_csv(tmp_path, labels=["Car", "Van", "Truck"])

    with open(tmp_path / "dets.csv", "r") as f:
        output = f.read()

    with open(tmp_path / "labels.txt", "r") as f:
        output_labels = f.read()

    with open(data_dir / "csv_save" / "dets.csv", "r") as f:
        exp_output = f.read()

    with open(data_dir / "csv_save" / "labels.txt", "r") as f:
        exp_output_labels = f.read()

    assert output == exp_output
    assert output_labels == exp_output_labels


def test_export_empty_csv(tmp_path: pathlib.Path, data_dir: pathlib.Path):
    empty_tracks = Tracks()
    empty_tracks.to_csv(tmp_path, labels=["Car", "Van", "Truck"])

    with open(tmp_path / "dets.csv", "r") as f:
        output = f.read()

    with open(tmp_path / "labels.txt", "r") as f:
        output_labels = f.read()

    with open(data_dir / "csv_save" / "labels.txt", "r") as f:
        exp_output_labels = f.read()

    assert output == ""
    assert output_labels == exp_output_labels


def test_export_csv_too_few_labels(sample_tracks: Tracks):
    with pytest.raises(ValueError, match="The length of provied labels"):
        sample_tracks.to_csv("kek", labels=["Car", "Van"])


def test_export_cvat_too_few_labels(sample_tracks: Tracks):
    with pytest.raises(ValueError, match="The length of provied labels"):
        sample_tracks.to_cvat_video("kek.xml", labels=["Car", "Van"], image_size=(1, 1))


def test_export_normal_cvat_video(
    sample_tracks: Tracks, tmp_path: pathlib.Path, data_dir: pathlib.Path
):
    with freezegun.freeze_time(
        dt.datetime.fromisoformat("2022-08-13T15:50:32.904197").isoformat()
    ):
        sample_tracks.to_cvat_video(
            tmp_path / "out.xml", labels=["Car", "Van", "Truck"], image_size=(640, 480)
        )

    with open(tmp_path / "out.xml", "r") as f:
        output = f.read()

    with open(data_dir / "cvat_video_export.xml", "r") as f:
        exp_output = f.read()

    assert output == exp_output


def test_export_empty_cvat_video(tmp_path: pathlib.Path, data_dir: pathlib.Path):
    empty_tracks = Tracks()

    with freezegun.freeze_time(
        dt.datetime.fromisoformat("2022-08-13T15:50:32.904197").isoformat()
    ):
        empty_tracks.to_cvat_video(
            tmp_path / "out.xml", labels=["Car", "Van", "Truck"], image_size=(640, 480)
        )

    with open(tmp_path / "out.xml", "r") as f:
        output = f.read()

    with open(data_dir / "cvat_video_export_empty.xml", "r") as f:
        exp_output = f.read()

    assert output == exp_output
