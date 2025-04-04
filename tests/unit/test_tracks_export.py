import datetime as dt
import pathlib

import freezegun
import pyarrow.parquet as pq
import pytest

from evaldet import Tracks


def test_export_normal_csv(
    sample_tracks: Tracks, tmp_path: pathlib.Path, data_dir: pathlib.Path
) -> None:
    sample_tracks.to_csv(tmp_path, labels=["Car", "Van", "Truck"])

    with open(tmp_path / "dets.csv") as f:
        output = f.read()

    with open(tmp_path / "labels.txt") as f:
        output_labels = f.read()

    with open(data_dir / "csv_save" / "dets.csv") as f:
        exp_output = f.read()

    with open(data_dir / "csv_save" / "labels.txt") as f:
        exp_output_labels = f.read()

    assert output == exp_output
    assert output_labels == exp_output_labels


def test_export_empty_csv(tmp_path: pathlib.Path, data_dir: pathlib.Path) -> None:
    empty_tracks = Tracks([], [], [])
    empty_tracks.to_csv(tmp_path, labels=["Car", "Van", "Truck"])

    with open(tmp_path / "dets.csv") as f:
        output = f.read()

    with open(tmp_path / "labels.txt") as f:
        output_labels = f.read()

    with open(data_dir / "csv_save" / "labels.txt") as f:
        exp_output_labels = f.read()

    assert output == ""
    assert output_labels == exp_output_labels


def test_export_csv_too_few_labels(sample_tracks: Tracks) -> None:
    with pytest.raises(ValueError, match="The length of provied labels"):
        sample_tracks.to_csv("kek", labels=["Car", "Van"])


def test_export_cvat_too_few_labels(sample_tracks: Tracks) -> None:
    with pytest.raises(ValueError, match="The length of provied labels"):
        sample_tracks.to_cvat_video("kek.xml", labels=["Car", "Van"], image_size=(1, 1))


def test_export_normal_cvat_video(
    sample_tracks: Tracks, tmp_path: pathlib.Path, data_dir: pathlib.Path
) -> None:
    with freezegun.freeze_time(
        dt.datetime.fromisoformat("2022-08-13T15:50:32.904197").isoformat()
    ):
        sample_tracks.to_cvat_video(
            tmp_path / "out.xml", labels=["Car", "Van", "Truck"], image_size=(640, 480)
        )

    with open(tmp_path / "out.xml") as f:
        output = f.read()

    with open(data_dir / "cvat_video_export.xml") as f:
        exp_output = f.read()

    assert output == exp_output


def test_export_empty_cvat_video(
    tmp_path: pathlib.Path, data_dir: pathlib.Path
) -> None:
    empty_tracks = Tracks([], [], [])

    with freezegun.freeze_time(
        dt.datetime.fromisoformat("2022-08-13T15:50:32.904197").isoformat()
    ):
        empty_tracks.to_cvat_video(
            tmp_path / "out.xml", labels=["Car", "Van", "Truck"], image_size=(640, 480)
        )

    with open(tmp_path / "out.xml") as f:
        output = f.read()

    with open(data_dir / "cvat_video_export_empty.xml") as f:
        exp_output = f.read()

    assert output == exp_output


def test_export_normal_parquet(
    sample_tracks: Tracks, tmp_path: pathlib.Path, data_dir: pathlib.Path
) -> None:
    sample_tracks.to_parquet(tmp_path / "out.parquet")

    table = pq.read_table(tmp_path / "out.parquet")
    exp_table = pq.read_table(data_dir / "out.parquet")

    assert table == exp_table


def test_export_empty_parquet(tmp_path: pathlib.Path, data_dir: pathlib.Path) -> None:
    empty_tracks = Tracks([], [], [])
    empty_tracks.to_parquet(tmp_path / "empty.parquet")

    table = pq.read_table(tmp_path / "empty.parquet")
    exp_table = pq.read_table(data_dir / "empty.parquet")

    assert table == exp_table
