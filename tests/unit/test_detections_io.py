from pathlib import Path

import numpy.testing as npt

from evaldet.detections import Detections


######################################################
# Parquet
######################################################

def test_export_empty_parquet(tmp_path: Path) -> None:
    dets = Detections([], [], [])
    dets.to_parquet(tmp_path / "dets.parquet")

    dets_r = Detections.from_parquet(tmp_path / "dets.parquet")

    assert dets_r.num_classes == 0
    assert dets_r.num_dets == 0
    assert dets_r.num_images == 0


def test_export_sample_parquet(tmp_path: Path, sample_detections: Detections) -> None:
    sample_detections.to_parquet(tmp_path / "dets.parquet")
    dets_r = Detections.from_parquet(tmp_path / "dets.parquet")

    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names


def test_export_sample_no_names_parquet(
    tmp_path: Path, sample_detections: Detections
) -> None:
    sample_detections._class_names = None
    sample_detections._image_names = None

    sample_detections.to_parquet(tmp_path / "dets.parquet")
    dets_r = Detections.from_parquet(tmp_path / "dets.parquet")

    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names


def test_export_sample_no_confs_parquet(
    tmp_path: Path, sample_detections: Detections
) -> None:
    sample_detections._confs = None

    sample_detections.to_parquet(tmp_path / "dets.parquet")
    dets_r = Detections.from_parquet(tmp_path / "dets.parquet")

    assert dets_r.confs is None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names


def test_open_parquet(data_dir: Path, sample_detections: Detections) -> None:
    dets_r = Detections.from_parquet(data_dir / "detections" / "sample.parquet")
    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names

######################################################
# COCO
######################################################

def test_export_empty_coco(tmp_path: Path) -> None:
    dets = Detections([], [], [])
    dets.to_coco(tmp_path / "dets.json")

    dets_r = Detections.from_coco(tmp_path / "dets.json")

    assert dets_r.num_classes == 0
    assert dets_r.num_dets == 0
    assert dets_r.num_images == 0


def test_export_sample_coco(tmp_path: Path, sample_detections: Detections) -> None:
    sample_detections.to_coco(tmp_path / "dets.json")
    dets_r = Detections.from_coco(tmp_path / "dets.json")

    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names


def test_export_sample_no_names_coco(
    tmp_path: Path, sample_detections: Detections
) -> None:
    sample_detections._class_names = None
    sample_detections._image_names = None

    sample_detections.to_coco(tmp_path / "dets.json")
    dets_r = Detections.from_coco(tmp_path / "dets.json")

    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert dets_r.class_names == ('0', '1', '2')
    assert dets_r.image_names == ('0', '1', '2')


def test_export_sample_no_confs_coco(
    tmp_path: Path, sample_detections: Detections
) -> None:
    sample_detections._confs = None

    sample_detections.to_coco(tmp_path / "dets.json")
    dets_r = Detections.from_coco(tmp_path / "dets.json")

    assert dets_r.confs is None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names


def test_open_coco(data_dir: Path, sample_detections: Detections) -> None:
    dets_r = Detections.from_coco(data_dir / "detections" / "coco.json")
    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)
    npt.assert_array_equal(sample_detections.confs, dets_r.confs)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names
