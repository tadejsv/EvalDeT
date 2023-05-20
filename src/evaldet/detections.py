from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt


class Detections:
    _image_ids: npt.NDArray[np.int32]
    _bboxes: npt.NDArray[np.float32]
    _classes: npt.NDArray[np.int32]
    _confs: Optional[npt.NDArray[np.float32]]
    _image_ind_dict: dict[int, tuple[int, int]]

    _class_names: Optional[tuple[str]]
    _image_names: Optional[tuple[str]]

    def __init__(
        self,
        image_ids: Union[npt.NDArray[np.int32], Sequence[int]],
        bboxes: Union[Sequence[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
        classes: Union[npt.NDArray[np.int32], Sequence[int]],
        confs: Optional[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        class_names: Optional[Sequence[str]] = None,
        image_names: Optional[Sequence[str]] = None,
    ) -> None:
        if len(image_ids) != len(bboxes):
            raise ValueError(
                "`detections` and `ids` should contain the same number of items."
            )

        if len(image_ids) != len(classes):
            raise ValueError(
                "`ids` and `frame_nums` should contain the same number of items."
            )

        if confs is not None and len(confs) != len(image_ids):
            raise ValueError(
                "If `classes` is given, it should contain the same number of items"
                " as `ids`."
            )

        if len(bboxes) > 0 and bboxes[0].shape != (4,):
            raise ValueError(
                "Each row of `bboxes` should be an 4-item array, but got"
                f" shape {bboxes[0].shape}"
            )

        if len(image_ids) == 0:
            self._image_ids = np.zeros((0,), dtype=np.int32)
            self._bboxes = np.zeros((0, 4), dtype=np.float32)
            self._classes = np.zeros((0,), dtype=np.int32)
            self._confs = np.zeros((0,), dtype=np.float32)

        else:
            image_ids = np.array(image_ids)
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

        # Check that class_names cover
        if class_names is None:
            self._class_names = None
        else:
            pass

        if image_names is None:
            self._image_names = None
        else:
            pass

        self._create_image_ind_dict()

    def _create_image_ind_dict(self) -> None:
        if len(self._frame_nums) == 0:
            self._frame_ind_dict = {}
            return

        u_frame_nums, start_inds = np.unique(self._frame_nums, return_index=True)
        frame_start_inds = start_inds.tolist()
        frame_end_inds = start_inds[1:].tolist() + [len(self._frame_nums)]

        frame_start_end_inds = zip(frame_start_inds, frame_end_inds)
        self._frame_ind_dict = dict(zip(u_frame_nums.tolist(), frame_start_end_inds))

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
    def image_ind_dict(self) -> dict[int, tuple[int, int]]:
        return self._image_ind_dict

    @property
    def class_names(self) -> Optional[tuple[str]]:
        return self._class_names

    @property
    def image_names(self) -> Optional[tuple[str]]:
        return self._image_names
