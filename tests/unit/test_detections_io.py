from pathlib import Path

import numpy as np
import numpy.testing as npt

from evaldet.detections import Detections

######################################################
# Parquet
######################################################


def test_export_empty_parquet(tmp_path: Path) -> None:
    dets = Detections([], [], [], class_names=("cls",), image_names=())
    dets.to_parquet(tmp_path / "dets.parquet")

    dets_r = Detections.from_parquet(tmp_path / "dets.parquet")

    assert dets_r.num_classes == 0
    assert dets_r.num_dets == 0
    assert dets_r.num_images == 0

    # len == 0 path in __init__ creates a 0-length iscrowd array
    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.zeros((0,), dtype=np.bool_))


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

    assert dets_r.iscrowd is None


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

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names

    assert dets_r.iscrowd is None


def test_open_parquet(data_dir: Path, sample_detections: Detections) -> None:
    dets_r = Detections.from_parquet(data_dir / "detections" / "sample.parquet")
    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names

    assert dets_r.iscrowd is None


######################################################
# COCO
######################################################


def test_export_empty_coco(tmp_path: Path) -> None:
    dets = Detections([], [], [], class_names=("cls",), image_names=())
    dets.to_coco(tmp_path / "dets.json")

    dets_r = Detections.from_coco(tmp_path / "dets.json")

    assert dets_r.num_classes == 0
    assert dets_r.num_dets == 0
    assert dets_r.num_images == 0

    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.zeros((0,), dtype=np.bool_))


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

    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.zeros((dets_r.num_dets,), dtype=np.bool_))


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

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names

    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.zeros((dets_r.num_dets,), dtype=np.bool_))


def test_open_coco(data_dir: Path, sample_detections: Detections) -> None:
    dets_r = Detections.from_coco(data_dir / "detections" / "coco.json")
    assert sample_detections.confs is not None
    assert dets_r.confs is not None

    npt.assert_array_equal(sample_detections.image_ids, dets_r.image_ids)
    npt.assert_array_equal(sample_detections.bboxes, dets_r.bboxes)
    npt.assert_array_equal(sample_detections.classes, dets_r.classes)

    assert sample_detections.class_names == dets_r.class_names
    assert sample_detections.image_names == dets_r.image_names

    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.zeros((dets_r.num_dets,), dtype=np.bool_))


def test_open_coco_nums(data_dir: Path) -> None:
    """Test what happens when image/category ids are not 0 indexed"""
    dets_r = Detections.from_coco(data_dir / "detections" / "coco_nums.json")
    assert dets_r.confs is not None

    npt.assert_array_equal(dets_r.image_ids, np.array([0], dtype=np.int32))
    npt.assert_array_equal(
        dets_r.bboxes, np.array([[0.0, 356.7, 76.6, 122.67]], dtype=np.float32)
    )
    npt.assert_array_equal(dets_r.classes, np.array([0], dtype=np.int32))
    npt.assert_array_equal(dets_r.confs, np.array([0.9], dtype=np.float32))

    assert dets_r.class_names == ("car",)
    assert dets_r.image_names == ("car3",)

    assert dets_r.iscrowd is not None
    npt.assert_array_equal(dets_r.iscrowd, np.array([False], dtype=np.bool_))


def test_open_coco_crowd(data_dir: Path) -> None:
    dets_r = Detections.from_coco(data_dir / "detections" / "coco_crowd.json")

    assert dets_r.confs is not None
    assert dets_r.iscrowd is not None

    # Expect image ids 0,1 and iscrowd True for first, False for second
    npt.assert_array_equal(dets_r.image_ids, np.array([0, 1], dtype=np.int32))
    npt.assert_array_equal(dets_r.classes, np.array([0, 0], dtype=np.int32))
    npt.assert_array_equal(dets_r.iscrowd, np.array([True, False], dtype=np.bool_))
