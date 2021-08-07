from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


class Tracks:
    """A class representing objects' tracks in a MOT setting.

    It allows for the tracks to be manually constructed, frame by frame,
    but also provides convenience class methods to initialize it from
    a file, the following formats are currently supported

    - MOT format (as described `here <https://motchallenge.net/instructions/>`__)
    - CVAT's version of the MOT format (as described `here <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/formats/format-mot/>`__)
    - CVAT for Video format (as described `here <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/xml_format/>`__)
    - UA-DETRAC XML format (you can download an example `here <https://detrac-db.rit.albany.edu/Tracking>`__)
    """

    @classmethod
    def from_mot(cls, file_path: Union[Path, str]):
        """Creates a Tracks object from detections file in the MOT format.

        The format should look like this::

            <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The values for ``conf``, ``x``, ``y`` and ``z`` will be
        ignored.

        Args:
            file_path: Path where the detections file is located. The file should be
                in the format described above, and should not have a header.
        """

        tracks = cls()

        with open(file_path, newline="") as file:
            fieldnames = [
                "frame",
                "id",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "_",
            ]
            csv_reader = csv.DictReader(file, fieldnames=fieldnames, dialect="unix")

            current_frame = -1
            detections: List[List[float]] = []
            ids: List[int] = []

            for line_num, line in enumerate(csv_reader):
                try:
                    frame_num, track_id = int(line["frame"]), int(line["id"])

                    xmin, ymin = float(line["bb_left"]), float(line["bb_top"])
                    xmax, ymax = xmin + float(line["bb_width"]), ymin + float(
                        line["bb_height"]
                    )
                except ValueError:
                    raise ValueError(
                        "Error when converting values to numbers on line"
                        f" {line_num}. Please check that all the values are numeric"
                        " and that the file follows the MOT format."
                    )

                if frame_num > current_frame:
                    # Flush current frame
                    if current_frame > -1:
                        tracks.add_frame(
                            current_frame,
                            ids=ids,
                            detections=np.array(detections, dtype=np.float32),
                        )

                    # Reset frame accumulator objects
                    current_frame = frame_num
                    detections = []
                    ids = []

                # Add to frame accumulator objects
                detections.append([xmin, ymin, xmax, ymax])
                ids.append(track_id)

            # Flush final frame
            tracks.add_frame(
                current_frame,
                ids=ids,
                detections=np.array(detections, dtype=np.float32),
            )

        return tracks

    @classmethod
    def from_mot_cvat(cls, file_path: Union[Path, str]) -> Tracks:
        """Creates a Tracks object from detections file in the CVAT's MOT format.

        The format should look like this::

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
                <frame density="2" num="2">
                    <target_list>
                        <target id="2">
                            <box height="32.44999999999999" left="329.27" top="96.65" width="56.53000000000003"/>
                            <attribute color="Multi" orientation="315" speed="1.0394" trajectory_length="3" truncation_ratio="0" vehicle_type="Car"/>
                        </target>
                        <target id="4">
                            <box height="122.67000000000002" left="0.0" top="356.7" width="76.6"/>
                            <attribute color="Multi" orientation="315" speed="1.0394" trajectory_length="1" truncation_ratio="0" vehicle_type="Car"/>
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

        if classes_attr_name and not classes_list:
            raise ValueError(
                "If you provide `classes_attr_name`,"
                " you must also provide `classes_list`"
            )

        xml_tree = ET.parse(file_path)
        root = xml_tree.getroot()
        tracks = cls()

        frames = root.findall("frame")
        for frame in frames:
            tracks_f = frame.find("target_list").findall("target")  # type: ignore

            current_frame = int(frame.attrib["num"])
            detections: List[List[float]] = []
            classes: List[int] = []
            ids: List[int] = []

            for track in tracks_f:
                # Get track attributes
                ids.append(int(track.attrib["id"]))

                box = track.find("box")
                xmin, ymin = float(box.attrib["left"]), float(box.attrib["top"])  # type: ignore
                xmax = xmin + float(box.attrib["width"])  # type: ignore
                ymax = ymin + float(box.attrib["height"])  # type: ignore
                detections.append([xmin, ymin, xmax, ymax])

                if classes_attr_name:
                    attrs = track.find("attribute")
                    class_val = attrs.attrib[classes_attr_name]  # type: ignore
                    class_val = classes_list.index(class_val)  # type: ignore
                    classes.append(class_val)

            tracks.add_frame(
                current_frame,
                ids=ids,
                detections=np.array(detections, dtype=np.float32),
                classes=classes if len(classes) else None,
            )

        return tracks

    @classmethod
    def from_cvat_video(
        cls,
        file_path: Union[Path, str],
        classes_list: List[str],
    ) -> Tracks:
        """Creates a Tracks object from detections file in the CVAT for Video XML
        format.

        Here's how this file might look like:

        .. code-block:: xml

            <annotations>
                <version>1.1</version>
                <meta>
                    <!-- lots of non-relevant metadata -->
                </meta>
                <track id="0" label="Car" source="manual">
                    <box frame="659" outside="0" occluded="0" keyframe="1" xtl="323.83" ytl="104.06" xbr="367.60" ybr="139.49" z_order="-1"> </box>
                    <box frame="660" outside="0" occluded="0" keyframe="1" xtl="320.98" ytl="105.24" xbr="365.65" ybr="140.95" z_order="0"> </box>
                </track>
                <track id="1" label="Car" source="manual">
                    <box frame="659" outside="0" occluded="0" keyframe="1" xtl="273.10" ytl="88.77" xbr="328.69" ybr="113.09" z_order="1"> </box>
                    <box frame="660" outside="0" occluded="0" keyframe="1" xtl="273.10" ytl="88.88" xbr="328.80" ybr="113.40" z_order="0"> </box>
                </track>
                <track id="2" label="Car" source="manual">
                    <box frame="659" outside="0" occluded="0" keyframe="1" xtl="375.24" ytl="80.43" xbr="401.65" ybr="102.67" z_order="0"> </box>
                    <box frame="660" outside="0" occluded="0" keyframe="1" xtl="374.69" ytl="80.78" xbr="401.09" ybr="103.01" z_order="0"> </box>
                </track>
                <track id="3" label="Car" source="manual">
                    <box frame="699" outside="0" occluded="0" keyframe="1" xtl="381.50" ytl="79.04" xbr="405.12" ybr="99.19" z_order="0"> </box>
                    <box frame="700" outside="0" occluded="0" keyframe="1" xtl="380.94" ytl="79.60" xbr="404.56" ybr="99.75" z_order="0"> </box>
                </track>
            </annotations>

        All attributes of each detection will be ignored, except for ``label`` (in the
        ``track`` object), which will be used for the ``class`` values. As this
        attribute usually contains string values, you also need to provide
        ``classes_list`` - a list of all possible class values. The class attribute will
        then be replaced by the index of the label in this list.

        .. warning::
            As this format organizes detections by id, instead of by frame, all
            detections must first be read in before they can be written to a
            ``Tracks`` object. Therefore, if you are opening large files, you
            can face large memory consumpition.

        Args:
            file_path: Path where the detections file is located
            classes_list: The list of all possible class values. The values from that
                attribute in the file will then be replaced by the index of that value
                in this list.
        """

        xml_tree = ET.parse(file_path)
        root = xml_tree.getroot()
        tracks = cls()

        frames: Dict[int, Any] = {}
        tracks_cvat = root.findall("track")
        for track_cvat in tracks_cvat:
            track_id = int(track_cvat.attrib["id"])
            track_class = classes_list.index(track_cvat.attrib["label"])

            for box in track_cvat.findall("box"):
                frame_num = int(box.attrib["frame"])
                xmin, ymin = float(box.attrib["xtl"]), float(box.attrib["ytl"])
                xmax, ymax = float(box.attrib["xbr"]), float(box.attrib["ybr"])

                current_frame = frames.get(frame_num, {})
                if current_frame:
                    current_frame["classes"].append(track_class)
                    current_frame["ids"].append(track_id)
                    current_frame["detections"].append([xmin, ymin, xmax, ymax])
                else:
                    current_frame["classes"] = [track_class]
                    current_frame["ids"] = [track_id]
                    current_frame["detections"] = [[xmin, ymin, xmax, ymax]]
                    frames[frame_num] = current_frame

        list_frames = sorted(list(frames.keys()))
        for frame in list_frames:
            tracks.add_frame(
                frame,
                ids=frames[frame]["ids"],
                detections=np.array(frames[frame]["detections"]),
                classes=frames[frame]["classes"],
            )

        return tracks

    def __init__(self):
        self._last_frame = -1

        self._frame_nums = []
        self._detections = dict()
        self._ids = dict()
        self._classes = dict()

        self._all_classes = set()
        self._ids_count = {}

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
            raise ValueError(
                f"The `ids` must be unique - got non-unique ids on frame {frame_num}"
            )

        # If all ok, add objects to collections
        self._detections[frame_num] = detections.copy()
        self._ids[frame_num] = ids.copy()

        for _id in ids:
            self._ids_count[_id] = self._ids_count.get(_id, 0) + 1

        if classes is not None:
            self._classes[frame_num] = classes.copy()
            self._all_classes.update(classes)

        self._last_frame = frame_num
        self._frame_nums.append(frame_num)

    @property
    def all_classes(self) -> Set[int]:
        """Get a set of all classes in the collection."""
        return self._all_classes.copy()

    @property
    def all_ids(self) -> Set[int]:
        """Get a set of all track ids in the collection."""
        return set(self._ids_count.keys())

    @property
    def ids_count(self) -> Dict[int, int]:
        """Get the number of frames that each id is present in.

        Returns:
            A dictionary where keys are the track ids, and values
            are the numbers of frames they appear in.
        """
        return self._ids_count.copy()

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
        """Get the frame with number ``idx``.

        Returns:
            A dictionary with the key ``'ids'``, ``'detections'``
            and ``'classes'`` (if available). The values are a list
            (for ids and classes) or a numpy array (for detections)
            with the values for each item in the frame.
        """
        if idx not in self:
            raise KeyError(f"The frame {idx} does not exist.")

        return_dict = {
            "ids": self._ids[idx],
            "detections": self._detections[idx],
        }
        if idx in self._classes:
            return_dict["classes"] = self._classes[idx]

        return return_dict
