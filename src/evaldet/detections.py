"""
Module defining a general class for holding object detections,
as well as reading from and converting to various format.
"""

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

_SEP = b"\x00"


class Detections:
    """
    A class representing object detections (with bounding boxes).

    It can read and save to the following file formats

    - YOLO format (as described [here](https://opencv.github.io/cvat/docs/manual/advanced/formats/format-yolo/))
    - COCO format (as described [here](https://cocodataset.org/#format-data))
    - PASCAL VOC format (as described [here](https://opencv.github.io/cvat/docs/manual/advanced/formats/format-voc/))
    - our custom efficient parquet format (see `from_parquet` and `to_parquet`)

    Internally, all the attributes are saved as a single numpy array, and sorted by
    image ids. This enables efficient storage while maintaining easy access to
    detections from individual frames.
    """

    _image_ids: npt.NDArray[np.int32]
    _bboxes: npt.NDArray[np.float32]
    _classes: npt.NDArray[np.int32]
    _confs: npt.NDArray[np.float32] | None
    _image_ind_dict: dict[int, tuple[int, int]]

    _class_names: tuple[str, ...]
    _image_names: tuple[str, ...]

    def __init__(
        self,
        image_ids: npt.NDArray[np.int32] | Sequence[int],
        bboxes: Sequence[npt.NDArray[np.float32]] | npt.NDArray[np.float32],
        classes: npt.NDArray[np.int32] | Sequence[int],
        class_names: Sequence[str],
        image_names: Sequence[str],
        confs: npt.NDArray[np.float32] | Sequence[float] | None = None,
    ) -> None:
        """
        Create a `Detections` instance.

        Args:
            image_ids: A sequence or array of image ids. It is assumed to be 0-indexed.
            bboxes: A sequence or array of detection bounding boxes, which should be
                in the format `xywh`, using a top-left-origin coordinate system.
            classes: A sequence or array of classes (labels), assumed to be 0-indexed.
            confs: An optional sequence or array of confidences (scores).
            class_names: A sequence (list, tuple) of class names. The `i`-th element in
                the sequence (zero indexed) is the class name for detections with class
                label `i`. The length should be larger than the max class label in
                `classes`.
            image_names: A sequence (list, tuple) of image names - this is used for
                matching ground truth and prediction detection for metrics. The `i`-th
                element in the sequence (zero indexed) is the image name for all images
                with the label `i`. The length should be larger than the max image label

        """
        if len(image_ids) != len(bboxes):
            raise ValueError(
                "`image_ids` and `bboxes` should contain the same number of items."
            )

        if len(image_ids) != len(classes):
            raise ValueError(
                "`image_ids` and `classes` should contain the same number of items."
            )

        if confs is not None and len(confs) != len(image_ids):
            msg = (
                "If `confs` is given, it should contain the same number of items"
                " as `image_ids`."
            )
            raise ValueError(msg)

        if len(bboxes) > 0 and bboxes[0].shape != (4,):
            msg = (
                "Each row of `bboxes` should be an 4-item array, but got"
                f" shape {bboxes[0].shape}"
            )
            raise ValueError(msg)

        if len(image_ids) == 0:
            self._image_ids = np.zeros((0,), dtype=np.int32)
            self._bboxes = np.zeros((0, 4), dtype=np.float32)
            self._classes = np.zeros((0,), dtype=np.int32)
            self._confs = np.zeros((0,), dtype=np.float32)

        else:
            image_ids = np.array(image_ids)

            if confs is not None:
                sort_inds = np.lexsort((confs, image_ids))  # type: ignore[arg-type]
            else:
                sort_inds = np.argsort(image_ids)

            self._image_ids = np.array(image_ids[sort_inds], copy=True, dtype=np.int32)

            self._bboxes = np.array(
                np.array(bboxes)[sort_inds], dtype=np.float32, copy=True
            )

            self._classes = np.array(
                np.array(classes)[sort_inds], dtype=np.int32, copy=True
            )

            if confs is None:
                self._confs = None
            else:
                self._confs = np.array(
                    np.array(confs)[sort_inds], dtype=np.float32, copy=True
                )

        # Check that class_names cover all classes
        num_classes = self._classes.max(initial=-1) + 1
        if len(class_names) < num_classes:
            msg = (
                f"The number of class names ({len(class_names)}) is less than"
                f" the number of classes in the data ({num_classes})"
            )
            raise ValueError(msg)

        self._class_names = tuple(class_names)

        # Check that image_names cover all images
        num_images = self._image_ids.max(initial=-1) + 1
        if len(image_names) < num_images:
            msg = (
                f"The number of image names ({len(image_names)}) is less than the"
                f"number of images in the data ({num_images})"
            )
            raise ValueError(msg)

        self._image_names = tuple(image_names)

        self._create_image_ind_dict()

    def _create_image_ind_dict(self) -> None:
        if len(self._image_ids) == 0:
            self._image_ind_dict = {}
            return

        u_image_ids, start_inds = np.unique(self._image_ids, return_index=True)
        image_start_inds = start_inds.tolist()
        image_end_inds = [*start_inds[1:].tolist(), len(self._image_ids)]

        image_start_end_inds = zip(image_start_inds, image_end_inds, strict=True)
        self._image_ind_dict = dict(
            zip(u_image_ids.tolist(), image_start_end_inds, strict=True)
        )

    @staticmethod
    def strs_to_bytes(strs: tuple[str, ...]) -> bytes:
        """
        Convert a tuple of strings into a single bytes object joined by the separator.
        """
        return _SEP.join(x.encode("utf8") for x in strs)

    @staticmethod
    def bytes_to_strs(strs: bytes) -> tuple[str, ...]:
        """
        Convert a bytes object, separated by the separator, back into a tuple of
        strings.
        """
        return tuple(x.decode("utf8") for x in strs.split(_SEP))

    @classmethod
    def from_coco(cls, file_path: str | Path) -> "Detections":
        """
        Read the detections from a json file in the COCO format.

        The file should have this structure:

        ```
        {
            "categories": [
                {
                    "id": 1,
                    "name": "cat"
                },
                ...
            ],
            "images": [
                {
                    "id": 1,
                    "file_name": "cat1.jpg"
                },
                ...
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [260, 177, 231, 199],
                    "score": 0.67
                },
            ]
        }
        ```

        Other information (like `info` and `licenses` sections, or information about
        `width` or `height` for `images`) can be present too, but it will be ignored.

        The `score` attribute of `annotations` is optional, but if present should be
        present on all annotations.

        Because `Detections` requires that images and classes are zero indexed, while
        in COCO there is no restriction on numbering of `id`s, the `id`s for `image`
        and `category` are not perserved - but the names and relationships are.

        Args:
            file_path: Path where the detections file is located

        """
        with open(file_path) as f:
            coco_dict: dict[str, Any] = json.load(f)

        images: list[dict[str, int | str]] = coco_dict["images"]
        images = sorted(images, key=lambda x: x["id"])

        categories: list[dict[str, int | str]] = coco_dict["categories"]
        annotations: list[dict[str, Any]] = coco_dict["annotations"]

        img_ind_dict = {im["id"]: i for i, im in enumerate(images)}
        cat_ind_dict = {cat["id"]: i for i, cat in enumerate(categories)}

        image_ids = np.zeros((len(annotations),), dtype=np.int32)
        classes = np.zeros((len(annotations),), dtype=np.int32)
        bboxes = np.zeros((len(annotations), 4), dtype=np.float32)

        if len(annotations) == 0 or "score" in annotations[0]:
            confs = np.zeros((len(annotations),), dtype=np.float32)
        else:
            confs = None

        for i, ann in enumerate(annotations):
            image_ids[i] = img_ind_dict[ann["image_id"]]
            classes[i] = cat_ind_dict[ann["category_id"]]
            bboxes[i] = ann["bbox"]

            if confs is not None:
                confs[i] = ann["score"]

        image_names: tuple[str, ...] = tuple(im["file_name"] for im in images)  # type: ignore[misc]
        class_names: tuple[str, ...] = tuple(cat["name"] for cat in categories)  # type: ignore[misc]

        return cls(
            image_ids=image_ids,
            bboxes=bboxes,
            confs=confs,
            classes=classes,
            class_names=class_names,
            image_names=image_names,
        )

    # @classmethod
    # def from_yolo(cls) -> "Detections":
    #     pass

    # @classmethod
    # def from_pascal_voc(cls) -> "Detections":
    #     pass

    @classmethod
    def from_parquet(cls, file_path: str | Path) -> "Detections":
        """
        Read the detections from a parquet file.

        The file should have the following columns:

        ```
        image_id, xmin, ymin, width, height, class
        ```

        Additionally, it can have a `conf` column. In metadata, it can provide the
        `class_names` and `image_names` attributes (concatenated names separated by a
        special character, as done in `to_parquet`).

        Args:
            file_path: Path where the detections file is located

        """
        table = pq.read_table(file_path)

        image_ids = table["image_id"].to_numpy()
        x = table["xmin"].to_numpy()
        y = table["ymin"].to_numpy()
        w = table["width"].to_numpy()
        h = table["height"].to_numpy()
        classes = table["class"].to_numpy()

        bboxes = np.stack([x, y, w, h]).T.copy()

        confs = table["conf"].to_numpy() if "conf" in table.column_names else None

        metadata = table.schema.metadata
        class_names = cls.bytes_to_strs(metadata[b"class_names"])
        image_names = cls.bytes_to_strs(metadata[b"image_names"])

        return cls(
            image_ids=image_ids,
            bboxes=bboxes,
            classes=classes,
            confs=confs,
            class_names=class_names,
            image_names=image_names,
        )

    def filter(self, mask: npt.NDArray[np.bool_]) -> "Detections":
        """
        Filter the detections using a boolean mask.

        This method will filter all attributes according to the mask provided.

        Args:
            mask: A boolean array, should be the same length as ids and other
                attributes.

        Return:
            The filtered detections

        """
        if mask.shape != self._image_ids.shape:
            msg = (
                "Shape of the filter should equal the shape of `image_ids`, got"
                f" {mask.shape}"
            )
            raise ValueError(msg)

        filtered_confs = self.confs if self.confs is None else self.confs[mask]

        return Detections(
            image_ids=self.image_ids[mask],
            bboxes=self.bboxes[mask],
            classes=self.classes[mask],
            confs=filtered_confs,
            class_names=self.class_names,
            image_names=self.image_names,
        )

    def to_coco(self, file_path: str | Path) -> None:
        """
        Export detections to COCO format.

        Args:
            file_path: Relative or absolute file name to save to.

        """
        if self.image_names is None:
            num_images = self.image_ids.max(initial=-1) + 1
            images = [{"id": i, "file_name": f"{i}"} for i in range(num_images)]
        else:
            images = [
                {"id": i, "file_name": name} for i, name in enumerate(self.image_names)
            ]

        if self.class_names is None:
            num_categories = self.classes.max(initial=-1) + 1
            categories = [{"id": int(i), "name": str(i)} for i in range(num_categories)]
        else:
            categories = [
                {"id": i, "name": name} for i, name in enumerate(self.class_names)
            ]

        annotations: list[dict[str, Any]] = []
        for i in range(len(self.bboxes)):
            ann: dict[str, Any] = {}
            ann["image_id"] = int(self.image_ids[i])
            ann["category_id"] = int(self.classes[i])
            ann["bbox"] = [round(x, 3) for x in self.bboxes[i].tolist()]

            if self.confs is not None:
                ann["score"] = round(float(self.confs[i]), 3)

            annotations.append(ann)

        coco_dict = {
            "licences": [],
            "info": {
                "year": dt.datetime.now().year,
                "date_created": dt.datetime.now().isoformat(timespec="seconds"),
            },
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

        with open(file_path, "w+") as f:
            json.dump(coco_dict, f)

    # def to_pascal_voc(self) -> None:
    #     pass

    # def to_yolo(self) -> None:
    #     pass

    def to_parquet(self, file_path: str | Path) -> None:
        """
        Export detections to parquet format.

        Args:
            file_path: Relative or absolute file name to save to.

        """
        columns: list[pa.Array] = [
            pa.array(self._image_ids),
            pa.array(self._bboxes[:, 0]),
            pa.array(self._bboxes[:, 1]),
            pa.array(self._bboxes[:, 2]),
            pa.array(self._bboxes[:, 3]),
            pa.array(self._classes),
        ]
        col_names: list[str] = ["image_id", "xmin", "ymin", "width", "height", "class"]

        if self._confs is not None:
            columns.append(pa.array(self._confs))
            col_names.append("conf")

        table = pa.Table.from_arrays(columns, col_names)

        metadata: dict[bytes, bytes] = {}
        if self._class_names is not None:
            metadata[b"class_names"] = self.strs_to_bytes(self._class_names)

        if self._image_names is not None:
            metadata[b"image_names"] = self.strs_to_bytes(self._image_names)

        table = table.cast(table.schema.with_metadata(metadata))
        pq.write_table(table, file_path)

    @property
    def num_images(self) -> int:
        """Number of images with detections."""
        return len(self._image_ind_dict)

    @property
    def num_dets(self) -> int:
        """Number of detections."""
        return len(self._bboxes)

    @property
    def num_classes(self) -> int:
        """Number of classes with detections."""
        return len(np.unique(self._classes))

    @property
    def image_ids(self) -> npt.NDArray[np.int32]:
        """The array of image IDs associated with each bounding box."""
        return self._image_ids

    @property
    def bboxes(self) -> npt.NDArray[np.float32]:
        """The array of bounding boxes, each defined by [x_min, y_min, x_max, y_max]."""
        return self._bboxes

    @property
    def classes(self) -> npt.NDArray[np.int32]:
        """The array of integer class labels for each bounding box."""
        return self._classes

    @property
    def confs(self) -> npt.NDArray[np.float32] | None:
        """The array of confidence scores if available, otherwise None."""
        return self._confs

    @property
    def image_ind_dict(self) -> dict[int, tuple[int, int]]:
        """
        A dictionary mapping image IDs to start and end indices for all
        bounding boxes belonging to the image.
        """
        return self._image_ind_dict

    @property
    def class_names(self) -> tuple[str, ...]:
        """A tuple of class names corresponding to the integer class labels."""
        return self._class_names

    @property
    def image_names(self) -> tuple[str, ...]:
        """A tuple of image names corresponding to the image IDs."""
        return self._image_names
