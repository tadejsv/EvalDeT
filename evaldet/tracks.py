import csv
import xml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


class Tracks:
    @classmethod
    def from_mot(cls, file_path: Union[Path, str]):
        pass

    @classmethod
    def from_mot_cvat(cls, file_path: Union[Path, str]):
        pass

    @classmethod
    def from_ua_detrac(cls, file_path: Union[Path, str]):
        pass

    def __init__(self):
        self._last_frame = -1

        self._frame_nums = []
        self._detections = []
        self._ids = []
        self._classes = []

        self._all_classes = set()

    def add_frame(
        self,
        frame_num: int,
        ids: List[int],
        detections: np.ndarray,
        classes: Optional[List[int]] = None,
    ):
        if frame_num <= self._last_frame:
            raise ValueError(
                f"You attempted to add frame {frame_num}, but frame {self._last_frame}"
                " is already in the collection. New frame numbers should be higher"
                " than the largest one in the collection."
            )

        if len(ids) == 0:
            raise ValueError(
                "You must pass a non-empty list of `ids` when adding a frame."
            )

        if detections.ndim != 2:
            raise ValueError("The `detections` must be a 2d numpy array.")

        if len(ids) != detections.shape[0]:
            raise ValueError(
                "The `detections` and `ids` should contain the same number of items."
            )

        if classes is not None and len(classes) != len(ids):
            raise ValueError(
                "If `classes` is given, it should contain the same number of items"
                " as `ids`."
            )

        if detections.shape[1] != 4:
            raise ValueError(
                "The `detections` should be an Nx4 array, but got"
                f" shape Nx{detections.shape[1]}"
            )

        if not (detections[:, 2] - detections[:, 0] > 0).all():
            raise ValueError(
                "Detections have to be in the format [xmin, ymin, xmax, ymax],"
                " but one of xmax values is smaller than or equal to its xmin value."
            )

        if not (detections[:, 3] - detections[:, 1] > 0).all():
            raise ValueError(
                "Detections have to be in the format [xmin, ymin, xmax, ymax],"
                " but one of ymax values is smaller than or equal to its ymin value."
            )

        if len(set(ids)) != len(ids):
            raise ValueError("The `ids` must be unique.")

        # If all ok, add objects to collections
        self._detections.append(detections.copy())
        self._ids.append(ids.copy())

        if classes is not None:
            self._classes.append(classes.copy())
            self._all_classes.update(classes)

        self._last_frame = frame_num
        self._frame_nums.append(frame_num)

    @property
    def all_classes(self) -> Set[Optional[int]]:
        return self._all_classes.copy()

    @property
    def frames(self) -> List[int]:
        return self._frame_nums.copy()

    def __len__(self) -> int:
        return len(self._frame_nums)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass
