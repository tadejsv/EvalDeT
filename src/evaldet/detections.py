from typing import Optional

import numpy as np
import numpy.typing as npt


class Detections:
    _image_ids: npt.NDArray[np.int32]
    _bboxes: npt.NDArray[np.float32]
    _classes: npt.NDArray[np.int32]
    _confs: Optional[npt.NDArray[np.float32]]
    _frame_ind_dict: dict[int, tuple[int, int]]

    _class_names: Optional[dict[int, str]]
    _image_names: Optional[dict[int, str]]

    def __init__(self) -> None:
        pass

    @classmethod
    def from_coco(cls) -> "Detections":
        pass

    @classmethod
    def from_yolo(cls) -> "Detections":
        pass

    @classmethod
    def from_pascal_voc(cls) -> "Detections":
        pass

    @classmethod
    def from_parquet(cls) -> "Detections":
        pass

    def filter_class(self) -> "Detections":
        pass

    def filter(self) -> "Detections":
        pass

    def to_coco(self) -> None:
        pass

    def to_pascal_voc(self) -> None:
        pass

    def to_yolo(self) -> None:
        pass

    def to_parquet(self) -> None:
        pass

    def num_images(self) -> int:
        pass

    def num_dets(self) -> int:
        pass

    @property
    def image_ids(self) -> npt.NDArray[np.int32]:
        return self._image_ids

    @property
    def bboxes(self) -> npt.NDArray[np.float32]:
        return self._bboxes

    @property
    def classes(self) -> npt.NDArray[np.int32]:
        return self._classes

    @property
    def confs(self) -> Optional[npt.NDArray[np.float32]]:
        return self._confs

    @property
    def frame_ind_dict(self) -> dict[int, tuple[int, int]]:
        return self._frame_ind_dict
