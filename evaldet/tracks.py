from __future__ import annotations

import csv
import xml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


class Tracks:
    """A class representing objects' tracks in a MOT setting.

    It allows for the tracks to be manually constructed, frame by frame,
    but also provides convenience class methods to initialize it from
    a file, the following formats are currently supported

    - MOT format (as described `here <https://motchallenge.net/instructions/>`_)
    - CVAT's version of the MOT format (as described `here <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/formats/format-mot/>`_)
    - UA-DETRAC XML format (you can download an example `here <https://detrac-db.rit.albany.edu/Tracking>`_)
    """

    @classmethod
    def from_mot(cls, file_path: Union[Path, str]):
        """Creates a Tracks object from detections file in the MOT format."""

        pass

    @classmethod
    def from_mot_cvat(cls, file_path: Union[Path, str]) -> Tracks:
        """Creates a Tracks object from detections file in the CVAT's MOT format.

        The format should look like this: ::

            <frame_id>, <track_id>, <x>, <y>, <w>, <h>, <not ignored>, <class_id>, <visibility>, <skipped>

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The last two elements (``visibility`` and ``skipped``) are
        optional. The values for ``not ignored``, ``visibility`` and ``skipped`` will be
        ignored.

        Args:
            file_path: Path where the detections file is located. The file should be
                in the format described above, and should not have a header.
        """

        tracks = cls()

        with open(file_path, newline="") as file:
            fieldnames = ["frame_id", "track_id", "x", "y", "w", "h", "_", "class_id"]
            csv_reader = csv.DictReader(file, fieldnames=fieldnames, dialect="unix")

            current_frame = -1
            detections: List[List[float]] = []
            classes: List[int] = []
            ids: List[int] = []

            for line_num, line in enumerate(csv_reader):
                try:
                    frame_num = int(line["frame_id"])
                    track_id, class_id = int(line["track_id"]), int(line["class_id"])

                    xmin, ymin = float(line["x"]), float(line["y"])
                    xmax, ymax = xmin + float(line["w"]), ymin + float(line["h"])
                except ValueError:
                    raise ValueError(
                        "Error when converting values to numbers on line"
                        f" {line_num}. Please check that all the values are numeric"
                        " and that the file follows the CVAT-MOT format."
                    )

                if frame_num > current_frame:
                    # Flush current frame
                    if current_frame > -1:
                        tracks.add_frame(
                            current_frame,
                            ids=ids,
                            detections=np.array(detections, dtype=np.float32),
                            classes=classes,
                        )

                    # Reset frame accumulator objects
                    current_frame = frame_num
                    detections = []
                    classes = []
                    ids = []

                # Add to frame accumulator objects
                detections.append([xmin, ymin, xmax, ymax])
                classes.append(class_id)
                ids.append(track_id)

            # Flush final frame
            tracks.add_frame(
                current_frame,
                ids=ids,
                detections=np.array(detections, dtype=np.float32),
                classes=classes,
            )

        return tracks

    @classmethod
    def from_ua_detrac(
        cls,
        file_path: Union[Path, str],
        classes_attr_name: Optional[str] = None,
        classes_list: Optional[List[str]] = None,
    ) -> Tracks:
        """Creates a Tracks object from detections file in the UA-DETRAC XML format.

        Here's how this file might look like:

        .. code-block:: xml

            <sequence name="MVI_20033">
                <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
                <ignored_region>
                    <box height="53.75" left="458.75" top="0.5" width="159.5"/>
                </ignored_region>
                <frame density="4" num="1">
                    <target_list>
                        <target id="1">
                            <box height="71.46" left="256.88" top="201.1" width="67.06"/>
                            <attribute color="Multi" orientation="315" speed="1.0394" trajectory_length="91" truncation_ratio="0" vehicle_type="Taxi"/>
                        </target>
                    </target_list>
                </frame>
            </sequence>

        The ``ignored_region`` node will not be taken into account - if you want
        some detections to be ignored, you need to filter them prior to the creation
        of the file.

        All attributes of each detection will be ignored, except for the one designated
        by ``classes_attr_name`` (for example, in original UA-DETRAC this could be
        ``"vehicle_type"``). This would then give values for ``classes`` attribute.
        As this attribute usually contains string values, you also need to provide
        ``classes_list`` - a list of all possible class values. The class attribute will
        then be replaced by the index of the label in this list.

        Args:
            file_path: Path where the detections file is located
            classes_attr_name: The name of the attribute to be used for the ``classes``
                attribute. If provided, ``classes_list`` must be provided as well.
            classes_list: The list of all possible class values - must be provided if
                ``classes_attr_name`` is provided. The values from that attribute in the
                file will then be replaced by the index of that value in this list.
        """

        pass

    def __init__(self):
        self._last_frame = -1

        self._frame_nums = []
        self._detections = dict()
        self._ids = dict()
        self._classes = dict()

        self._all_classes = set()

    def add_frame(
        self,
        frame_num: int,
        ids: List[int],
        detections: np.ndarray,
        classes: Optional[List[int]] = None,
    ):
        """Add a frame (observation) to the collection.

        Args:
            frame_num: A non-negative frame number, must be larger than
                the number of the most recently added frame.
            ids: A list with ids of the objects in the frame.
            detections: An Nx4 array describing the bounding boxes of objects in
                the frame. It should be in the ``[xmin, ymin, xmax, ymax]`` format.
            classes: An optional list of classes for the objects. If passed all
                objects in the frame must be assigned a class.
        """

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
        self._detections[frame_num] = detections.copy()
        self._ids[frame_num] = ids.copy()

        if classes is not None:
            self._classes[frame_num] = classes.copy()
            self._all_classes.update(classes)

        self._last_frame = frame_num
        self._frame_nums.append(frame_num)

    @property
    def all_classes(self) -> Set[Optional[int]]:
        """Get a set of all classes in the collection."""
        return self._all_classes.copy()

    @property
    def frames(self) -> List[int]:
        """Get an ordered list of all frame numbers in the collection."""
        return self._frame_nums.copy()

    def __len__(self) -> int:
        return len(self._frame_nums)

    def __contains__(self, idx: int) -> bool:
        """Whether the frame ``idx`` is present in the collection."""
        return idx in self._frame_nums

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get the frame with number ``idx``."""
        if idx not in self:
            raise KeyError(f"The frame {idx} does not exist.")

        return_dict = {
            "ids": self._ids[idx],
            "detections": self._detections[idx],
        }
        if idx in self._classes:
            return_dict["classes"] = self._classes[idx]

        return return_dict
