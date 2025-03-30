"""
Module defining a general class for holding object detections,
as well as reading from and converting to various format.
"""

import collections as co
import csv
import datetime as dt
import pathlib
import typing as t
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

_ID_KEY = "id"
_FRAME_KEY = "frame"
_CONF_KEY = "conf"
_CLASS_KEY = "class"
_XMIN_KEY = "xmin"
_YMIN_KEY = "ymin"
_WIDTH_KEY = "width"
_HEIGHT_KEY = "height"


class FrameTracks(t.NamedTuple):
    """A named tuple holding the attributes of tracks in a given frame."""

    bboxes: npt.NDArray[np.float32]
    ids: npt.NDArray[np.int32]
    classes: npt.NDArray[np.int32]
    confs: npt.NDArray[np.float32]


class Tracks:
    """
    A class representing objects' tracks in a MOT setting.

    It can read the following MOT file formats

    - MOT format (as described [here](https://motchallenge.net/instructions/))
    - MOT ground truth format (as described [here](https://arxiv.org/abs/1603.00831))
    - CVAT's version of the MOT format (as described [here](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/formats/format-mot/))
    - CVAT for Video format (as described [here](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/xml_format/))
    - UA-DETRAC XML format (you can download an example [here](https://detrac-db.rit.albany.edu/Tracking))

    Internally, all the attributes are saved as a single numpy array, and sorted by
    frame numbers. This enables easy access, as well as easy conversion to/from formats
    that do not store detections by frames (but by tracks).

    The frame numbers will be zero-indexed internally, so for the MOT files 1 will be
    subtracted from all frame numbers.
    """

    _frame_nums: npt.NDArray[np.int32]
    _ids: npt.NDArray[np.int32]
    _bboxes: npt.NDArray[np.float32]
    _classes: npt.NDArray[np.int32]
    _confs: npt.NDArray[np.float32]
    _frame_ind_dict: dict[int, tuple[int, int]]

    @classmethod
    def from_csv(
        cls,
        csv_file: str | pathlib.Path,
        fieldnames: t.Sequence[str],
        zero_indexed: bool = True,
    ) -> "Tracks":
        """
        Get detections from a CSV file.

        The CSV file should have a normal comma (,) as a separator, and should not
        include a header.

        Args:
            csv_file: path to the CSV file
            fieldnames: The names of the fields. This will be passed to
                `csv.DictReader`. It should contain the names of the fields, in order
                that they appear. The following names will be used (others will be
                disregarded):
                - `xmin`
                - `ymin`
                - `height`
                - `width`
                - `conf`: for the confidence of the item
                - `class`: for the class label of the item
                - `id`: for the id of the item
                - `frame`: for the frame number
            zero_indexed: If the frame numbers are zero indexed. Otherwise they are
                assumed to be 1 indexed, and 1 will be subtracted from all frame numbers
                to make them zero indexed.

        """
        with open(csv_file, newline="") as file:
            csv_reader = csv.DictReader(file, fieldnames=fieldnames, dialect="unix")
            accumulator: dict[str, list[t.Any]] = co.defaultdict(list)

            for line_num, line in enumerate(csv_reader):
                try:
                    cls._add_to_tracks_accumulator(accumulator, line)

                except ValueError as e:
                    msg = (
                        "Error when converting values to numbers on line"
                        f" {line_num}. Please check that all the values are numeric"
                        " and that the file follows the MOT format."
                    )
                    raise ValueError(msg) from e

        return cls(**accumulator, zero_indexed=zero_indexed)

    @classmethod
    def from_mot(cls, file_path: pathlib.Path | str) -> "Tracks":
        """
        Creates a Tracks object from detections file in the MOT format.

        The format should look like this

        ```
        <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <conf>, <x>, <y>, <z>
        ```

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The values for `x`, `y` and `z` will be ignored.

        The frame numbers will be zero-indexed internally, so 1 will be subtracted from
        all frame numbers.

        Args:
            file_path: Path where the detections file is located. The file should be
                in the format described above, and should not have a header.

        """
        fieldnames = [
            _FRAME_KEY,
            _ID_KEY,
            _XMIN_KEY,
            _YMIN_KEY,
            _WIDTH_KEY,
            _HEIGHT_KEY,
            _CONF_KEY,
            "_",
        ]

        return cls.from_csv(file_path, fieldnames, zero_indexed=False)

    @classmethod
    def from_mot_gt(cls, file_path: pathlib.Path | str) -> "Tracks":
        """
        Creates a Tracks object from detections file in the MOT ground truth format.
        This format has some more information compared to the normal.

        The format should look like this
        ```
        <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <conf>, <class>, <visibility>
        ```

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The value for `visibility` will be ignored.

        The frame numbers will be zero-indexed internally, so 1 will be subtracted from
        all frame numbers.

        Args:
            file_path: Path where the detections file is located. The file should be
                in the format described above, and should not have a header.

        """
        fieldnames = [
            _FRAME_KEY,
            _ID_KEY,
            _XMIN_KEY,
            _YMIN_KEY,
            _WIDTH_KEY,
            _HEIGHT_KEY,
            _CONF_KEY,
            _CLASS_KEY,
            "visibility",
        ]

        return cls.from_csv(file_path, fieldnames, zero_indexed=False)

    @classmethod
    def from_mot_cvat(cls, file_path: pathlib.Path | str) -> "Tracks":
        """
        Creates a Tracks object from detections file in the CVAT's MOT format.

        The format should look like this::

        ```
        <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <not ignored>, <class>, <visibility>, <skipped>
        ```

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The last two elements (`visibility` and `skipped`) are
        optional. The values for `not ignored`, `visibility` and `skipped` will be
        ignored.

        The frame numbers will be zero-indexed internally, so 1 will be subtracted from
        all frame numbers.

        Args:
            file_path: Path where the detections file is located. The file should be
                in the format described above, and should not have a header.

        """  # noqa: E501
        fieldnames = [
            _FRAME_KEY,
            _ID_KEY,
            _XMIN_KEY,
            _YMIN_KEY,
            _WIDTH_KEY,
            _HEIGHT_KEY,
            "_",
            _CLASS_KEY,
        ]

        return cls.from_csv(file_path, fieldnames, zero_indexed=False)

    @classmethod
    def from_ua_detrac(
        cls,
        file_path: pathlib.Path | str,
        classes_attr_name: str | None = None,
        classes_list: t.Sequence[str] | None = None,
    ) -> "Tracks":
        """
        Creates a Tracks object from detections file in the UA-DETRAC XML format.

        Here's how this file might look like:

        ```
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
        ```

        The `ignored_region` node will not be taken into account - if you want
        some detections to be ignored, you need to filter them prior to the creation
        of the file.

        All attributes of each detection will be ignored, except for the one designated
        by `classes_attr_name` (for example, in original UA-DETRAC this could be
        `"vehicle_type"`). This would then give values for `classes` attribute.
        As this attribute usually contains string values, you also need to provide
        `classes_list` - a list of all possible class values. The class attribute will
        then be replaced by the index of the label in this list.

        Args:
            file_path: Path where the detections file is located
            classes_attr_name: The name of the attribute to be used for the `classes`
                attribute. If provided, `classes_list` must be provided as well.
            classes_list: The list of all possible class values - must be provided if
                `classes_attr_name` is provided. The values from that attribute in the
                file will then be replaced by the index of that value in this list.

        """  # noqa: E501
        if classes_attr_name and not classes_list:
            msg = (
                "If you provide `classes_attr_name`,"
                " you must also provide `classes_list`"
            )
            raise ValueError(msg)

        xml_tree = ET.parse(file_path)  # noqa: S314
        root = xml_tree.getroot()

        accumulator: dict[str, list[t.Any]] = co.defaultdict(list)

        frames = root.findall("frame")
        for frame in frames:
            frame_targets = frame.find("target_list")
            if frame_targets is None:
                raise ValueError("frame_targets (target_list) should not be None.")
            tracks_f = frame_targets.findall("target")

            current_frame = frame.attrib["num"]
            for track in tracks_f:
                box = track.find("box")
                if box is None:
                    raise ValueError("box should not be None")

                det_item = {
                    _ID_KEY: track.attrib["id"],
                    _FRAME_KEY: current_frame,
                    _HEIGHT_KEY: box.attrib["height"],
                    _WIDTH_KEY: box.attrib["width"],
                    _XMIN_KEY: box.attrib["left"],
                    _YMIN_KEY: box.attrib["top"],
                }

                if classes_attr_name:
                    attrs = track.find("attribute")
                    if attrs is None:
                        raise ValueError("attrs should not be None")
                    if classes_list is None:
                        raise ValueError("classes_list should not be None")

                    class_val = attrs.attrib[classes_attr_name]
                    det_item[_CLASS_KEY] = classes_list.index(class_val)

                cls._add_to_tracks_accumulator(accumulator, det_item)

        return cls(**accumulator, zero_indexed=True)

    @classmethod
    def from_cvat_video(
        cls,
        file_path: pathlib.Path | str,
        classes_list: t.Sequence[str],
    ) -> "Tracks":
        """
        Creates a Tracks object from detections file in the CVAT for Video XML format.

        Here's how this file might look like:

        ```xml
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
        ```

        All attributes of each detection will be ignored, except for `label` (in the
        `track` object), which will be used for the `class` values. As this
        attribute usually contains string values, you also need to provide
        `classes_list` - a list of all possible class values. The class attribute will
        then be replaced by the index of the label in this list.

        Elements with `"outside=1"` will be ignored.

        Args:
            file_path: Path where the detections file is located
            classes_list: The list of all possible class values. The values from that
                attribute in the file will then be replaced by the index of that value
                in this list.

        """  # noqa: E501
        xml_tree = ET.parse(file_path)  # noqa: S314
        root = xml_tree.getroot()
        accumulator: dict[str, list[t.Any]] = co.defaultdict(list)

        tracks_cvat = root.findall("track")
        for track_cvat in tracks_cvat:
            track_id = track_cvat.attrib["id"]
            track_class = classes_list.index(track_cvat.attrib["label"])

            for box in track_cvat.findall("box"):
                if int(box.attrib["outside"]) == 1:
                    continue

                det_item = {
                    _ID_KEY: track_id,
                    _FRAME_KEY: box.attrib["frame"],
                    _HEIGHT_KEY: float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
                    _WIDTH_KEY: float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                    _XMIN_KEY: box.attrib["xtl"],
                    _YMIN_KEY: box.attrib["ytl"],
                    _CLASS_KEY: track_class,
                }

                cls._add_to_tracks_accumulator(accumulator, det_item)

        return cls(**accumulator, zero_indexed=True)

    @classmethod
    def from_parquet(cls, file_path: pathlib.Path | str) -> "Tracks":
        """
        Read the tracks from a parquet file.

        The file should have the following columns:

        ```
        <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <conf>, <class>
        ```

        Args:
            file_path: Path where the detections file is located

        """
        table = pq.read_table(file_path)
        table_cols = table.column_names
        return cls(
            ids=table[_ID_KEY].to_numpy(),
            frame_nums=table[_FRAME_KEY].to_numpy(),
            bboxes=np.stack(
                (
                    table[_XMIN_KEY].to_numpy(),
                    table[_YMIN_KEY].to_numpy(),
                    table[_WIDTH_KEY].to_numpy(),
                    table[_HEIGHT_KEY].to_numpy(),
                ),
                axis=-1,
            ),
            classes=table[_CLASS_KEY] if _CLASS_KEY in table_cols else None,
            confs=table[_CONF_KEY] if _CONF_KEY in table_cols else None,
        )

    @staticmethod
    def _add_to_tracks_accumulator(
        accumulator: dict[str, t.Any], new_obj: dict[str, t.Any]
    ) -> None:
        frame_num, track_id = int(new_obj[_FRAME_KEY]), int(float(new_obj[_ID_KEY]))

        xmin, ymin = float(new_obj[_XMIN_KEY]), float(new_obj[_YMIN_KEY])
        width, height = float(new_obj[_WIDTH_KEY]), float(new_obj[_HEIGHT_KEY])

        accumulator["frame_nums"].append(frame_num)
        accumulator["ids"].append(track_id)
        accumulator["bboxes"].append(
            np.array([xmin, ymin, width, height], dtype=np.float32)
        )

        if _CONF_KEY in new_obj:
            conf = float(new_obj[_CONF_KEY])
            accumulator["confs"].append(conf)
        if _CLASS_KEY in new_obj:
            class_id = int(new_obj[_CLASS_KEY])
            accumulator["classes"].append(class_id)

    def __init__(
        self,
        ids: t.Sequence[int] | npt.NDArray[np.int32],
        frame_nums: t.Sequence[int] | npt.NDArray[np.int32],
        bboxes: t.Sequence[npt.NDArray[np.float32]] | npt.NDArray[np.float32],
        classes: t.Sequence[int] | npt.NDArray[np.int32] | None = None,
        confs: t.Sequence[float] | npt.NDArray[np.float32] | None = None,
        zero_indexed: bool = True,
    ) -> None:
        """
        Create a `Tracks` instance.

        Args:
            ids: A list or array of track ids.
            frame_nums: A list or array of frame numbers.
            bboxes: A list or array of detection bounding boxes, which should be
                in the format `xywh`, using a top-left-origin coordinate system.
            classes: A list or array of classes (labels). It can not be provided, in
                this case internally all detections will have a class of ``0``.
            confs: A list or array of confidences (scores). It can not be provided, in
                this case internally all detections will have a confidence of ``1``.
            zero_indexed: If the frame numbers are zero indexed. If not, it is assumed
                that they are 1-indexed - that is, they start with 1, and will be
                transformed to 0-indexed internally, by subtracting 1 from them.

        """
        if len(ids) != len(bboxes):
            msg = "`bboxes` and `ids` should contain the same number of items."
            raise ValueError(msg)

        if len(ids) != len(frame_nums):
            msg = "`ids` and `frame_nums` should contain the same number of items."
            raise ValueError(msg)

        if classes is not None and len(classes) != len(ids):
            msg = (
                "If `classes` is given, it should contain the same number of items"
                " as `ids`."
            )
            raise ValueError(msg)

        if confs is not None and len(confs) != len(ids):
            msg = (
                "If `confs` is given, it should contain the same number of items"
                " as `ids`."
            )
            raise ValueError(msg)

        if len(bboxes) > 0 and bboxes[0].shape != (4,):
            msg = (
                "Each row of `bboxes` should be an 4-item array, but got"
                f" shape {bboxes[0].shape}"
            )
            raise ValueError(msg)

        if len(ids) == 0:
            self._frame_nums = np.zeros((0,), dtype=np.int32)
            self._ids = np.zeros((0,), dtype=np.int32)
            self._bboxes = np.zeros((0, 4), dtype=np.float32)
            self._classes = np.zeros((0,), dtype=np.int32)
            self._confs = np.zeros((0,), dtype=np.float32)

        else:
            frame_nums = np.array(frame_nums)
            sort_inds = np.argsort(frame_nums)

            self._frame_nums = np.array(
                frame_nums[sort_inds], copy=True, dtype=np.int32
            )

            self._ids = np.array(np.array(ids)[sort_inds], dtype=np.int32, copy=True)
            self._bboxes = np.array(
                np.array(bboxes)[sort_inds], dtype=np.float32, copy=True
            )

            if zero_indexed is False:
                self._frame_nums -= 1

            if classes is None:
                self._classes = np.zeros((len(ids),), dtype=np.int32)
            else:
                self._classes = np.array(
                    np.array(classes)[sort_inds], dtype=np.int32, copy=True
                )

            if confs is None:
                self._confs = np.ones((len(ids),), dtype=np.float32)
            else:
                self._confs = np.array(
                    np.array(confs)[sort_inds], dtype=np.float32, copy=True
                )

        self._create_frame_ind_dict()

    def _create_frame_ind_dict(self) -> None:
        if len(self._frame_nums) == 0:
            self._frame_ind_dict = {}
            return

        u_frame_nums, start_inds = np.unique(self._frame_nums, return_index=True)
        frame_start_inds = start_inds.tolist()
        frame_end_inds = [*start_inds[1:].tolist(), len(self._frame_nums)]

        frame_start_end_inds = zip(frame_start_inds, frame_end_inds, strict=True)
        self._frame_ind_dict = dict(
            zip(u_frame_nums.tolist(), frame_start_end_inds, strict=True)
        )

    def filter(self, mask: npt.NDArray[np.bool]) -> "Tracks":
        """
        Filter the tracks using a boolean mask.

        This method will filter all attributes according to the mask provided.

        Args:
            mask: A boolean array, should be the same length as ids and other
                attributes.

        Return:
            The filtered tracks

        """
        if mask.shape != self.ids.shape:
            msg = f"Shape of the mask should equal the shape of ids, got {mask.shape}"
            raise ValueError(msg)

        return Tracks(
            ids=self._ids[mask],
            frame_nums=self._frame_nums[mask],
            bboxes=self._bboxes[mask],
            classes=self._classes[mask],
            confs=self._confs[mask],
        )

    @property
    def ids(self) -> npt.NDArray[np.int32]:
        """The array of ids of tracks for all detections."""
        return self._ids

    @property
    def frame_nums(self) -> npt.NDArray[np.int32]:
        """The array of frame numbers (indices) for all detections."""
        return self._frame_nums

    @property
    def bboxes(self) -> npt.NDArray[np.float32]:
        """The array of bounding boxes for all detections."""
        return self._bboxes

    @property
    def classes(self) -> npt.NDArray[np.int32]:
        """The array of classes for all detections."""
        return self._classes

    @property
    def confs(self) -> npt.NDArray[np.float32]:
        """The array of confidence scores for all detections."""
        return self._confs

    @property
    def all_classes(self) -> set[int]:
        """Get a set of all classes in the collection."""
        return set(np.unique(self._classes).tolist())

    @property
    def ids_count(self) -> dict[int, int]:
        """
        Get the number of frames that each id is present in.

        Returns:
            A dictionary where keys are the track ids, and values
            are the numbers of frames they appear in.

        """
        ids, counts = np.unique(self._ids, return_counts=True)
        return dict(zip(ids.tolist(), counts.tolist(), strict=True))

    @property
    def frames(self) -> set[int]:
        """Get an ordered list of all frame numbers in the collection."""
        return set(self._frame_ind_dict.keys())

    def __len__(self) -> int:
        """Get the number of tracks."""
        return len(self._ids)

    def __contains__(self, idx: int) -> bool:
        """Whether the frame `idx` is present in the collection."""
        return idx in self._frame_ind_dict

    @t.overload
    def __getitem__(self, idx: int) -> FrameTracks:
        pass

    @t.overload
    def __getitem__(self, idx: slice) -> "Tracks":
        pass

    def __getitem__(self, idx: int | slice) -> t.Union[FrameTracks, "Tracks"]:
        """
        Get a specific frame, or select only a subset of frames.

        Note that the `step` argument is not supported and will result in an error
        being raised if it is supplied. Negative indices for start or stop argument are
        similarly not supported.

        Args:
            idx: Can be either a non-negative integer - in this case, the specific frame
                will be returned, or a slice, in which case a tracks will be returned
                for the subset of frames defined by the slice.

        """
        if isinstance(idx, int):
            if idx < 0:
                msg = "Indexing with negative values is not supported."
                raise ValueError(msg)
            if idx not in self:
                msg = f"The frame {idx} does not exist."
                raise KeyError(msg)

            start, end = self._frame_ind_dict[idx]
            return FrameTracks(
                ids=self._ids[start:end],
                bboxes=self._bboxes[start:end],
                classes=self._classes[start:end],
                confs=self._confs[start:end],
            )

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if step is not None:
                raise ValueError("Slicing with the step argument is not supported.")
            if start < 0 or stop < 0:
                raise ValueError("Slicing with negative indices is not supported.")

            keep_frames = self.frames.intersection(list(range(start, stop)))
            if len(keep_frames) == 0:
                return type(self)([], [], [])

            start = self._frame_ind_dict[min(keep_frames)][0]
            end = self._frame_ind_dict[max(keep_frames)][1]

            return type(self)(
                ids=self._ids[start:end],
                frame_nums=self._frame_nums[start:end],
                bboxes=self._bboxes[start:end],
                classes=self._classes[start:end],
                confs=self._confs[start:end],
            )

        msg = "Unrecognized index type"
        raise ValueError(msg)

    def to_cvat_video(  # noqa: PLR0915
        self,
        filename: pathlib.Path | str,
        labels: t.Sequence[str],
        image_size: tuple[int, int] = (1, 1),
    ) -> None:
        """
        Export detections to CVAT for Video 1.1 format.

        More information on the format can be found
        [here](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/).

        Args:
            filename: The name of the file to save to - should have an `.xml` suffix.
            labels: A list/tuple of label names. The length should be at least the
                maximum label index - 1 (the first label corresponds to label at the
                0-th index).
            image_size: The size of the image in the `[w, h]` format, in pixels.

        """
        max_label_ind = max(self.all_classes, default=0) + 1
        if len(labels) < max_label_ind:
            msg = (
                f"The length of provied labels {len(labels)} is less than the largest"
                f" label id (+1) among tracklets {max_label_ind}."
            )
            raise ValueError(msg)

        w, h = image_size
        max_frame = max(self.frames, default=1)

        annotations = ET.Element("annotations")
        version = ET.SubElement(annotations, "version")
        version.text = "1.1"

        meta = ET.SubElement(annotations, "meta")

        dumped = ET.SubElement(meta, "dumped")
        now = dt.datetime.now().replace(tzinfo=None)
        dumped.text = now.isoformat().replace("T", " ")

        task = ET.SubElement(meta, "task")
        task_id = ET.SubElement(task, "id")
        task_id.text = "1"
        task_name = ET.SubElement(task, "name")
        task_name.text = "Tracking"
        task_size = ET.SubElement(task, "size")
        task_size.text = str(max_frame)
        task_mode = ET.SubElement(task, "mode")
        task_mode.text = "interpolation"
        ET.SubElement(task, "overlap")
        ET.SubElement(task, "bugtracker")
        task_flipped = ET.SubElement(task, "flipped")
        task_flipped.text = "False"
        task_created = ET.SubElement(task, "created")
        task_created.text = dumped.text
        task_updated = ET.SubElement(task, "updated")
        task_updated.text = dumped.text

        labels_el = ET.SubElement(task, "labels")
        for label_name in labels:
            label_el = ET.SubElement(labels_el, "label")
            label_name_el = ET.SubElement(label_el, "name")
            label_name_el.text = label_name
            ET.SubElement(label_el, "attributes")

        owner = ET.SubElement(task, "owner")
        ET.SubElement(owner, "username")
        ET.SubElement(owner, "email")

        original_size = ET.SubElement(task, "original_size")
        original_size_w = ET.SubElement(original_size, "width")
        original_size_h = ET.SubElement(original_size, "height")
        original_size_w.text = f"{w}"
        original_size_h.text = f"{h}"

        segments = ET.SubElement(task, "segments")
        segment = ET.SubElement(segments, "segment")
        segment_id = ET.SubElement(segment, "id")
        segment_id.text = "0"
        segment_start = ET.SubElement(segment, "start")
        segment_start.text = "0"
        segment_stop = ET.SubElement(segment, "stop")
        segment_stop.text = str(max_frame)
        ET.SubElement(segment, "url")

        for track_id in np.unique(self.ids):
            id_inds = np.nonzero(self.ids == track_id)[0]
            track_frames = self.frame_nums[id_inds]

            # Take first label, as only one is supported
            label: int = self.classes[id_inds[0]]

            track_el = ET.SubElement(
                annotations,
                "track",
                id=str(track_id),
                label=labels[label],
                source=type(self).__name__,
            )

            for i, (ind, frame_num) in enumerate(
                zip(id_inds, track_frames, strict=True)
            ):
                bbox = self.bboxes[ind]
                ET.SubElement(
                    track_el,
                    "box",
                    frame=str(frame_num),
                    xtl=f"{bbox[0]:.2f}",
                    ytl=f"{bbox[1]:.2f}",
                    xbr=f"{bbox[0] + bbox[2]:.2f}",
                    ybr=f"{bbox[1] + bbox[3]:.2f}",
                    outside="0",
                    occluded="0",
                    keyframe="1",
                )

                # Add fake element with outside=1 to prevent it showing up on CVAT
                if frame_num + 1 != track_frames[min(i + 1, len(track_frames) - 1)]:
                    ET.SubElement(
                        track_el,
                        "box",
                        frame=str(frame_num + 1),
                        xtl=f"{bbox[0]:.2f}",
                        ytl=f"{bbox[1]:.2f}",
                        xbr=f"{bbox[0] + bbox[2]:.2f}",
                        ybr=f"{bbox[1] + bbox[3]:.2f}",
                        outside="1",  # Make it outside to prevent ghost frames
                        occluded="0",
                        keyframe="1",
                    )

        tree = ET.ElementTree(annotations)
        tree.write(
            filename, xml_declaration=True, short_empty_elements=False, encoding="utf-8"
        )

    def to_csv(
        self,
        dirname: pathlib.Path | str,
        labels: t.Sequence[str],
    ) -> None:
        """
        Export detections to a simple CSV format.

        The format comprises of two files:
        `dets.csv`, containing the detections, and `labels.txt`, which contains
        the names of the labels (corresponding to label indices in `dets.csv`). The
        rows in `dets.csv` have the following format:

            <frame>, <id>, <x_min>, <y_min>, <width>, <height>, <class>, <conf>

        Note that `frame` and `class` are both 0 indexed.

        Args:
            dirname: The name of the directory to save to - will be created if it
                doesn't already exist.
            labels: A list/tuple of label names. The length should be at least the
                maximum label index - 1 (the first label corresponds to label at the
                0-th index).

        """
        max_label_ind = max(self.all_classes, default=0) + 1
        if len(labels) < max_label_ind:
            msg = (
                f"The length of provied labels {len(labels)} is less than the largest"
                f" label id (+1) among tracklets {max_label_ind}."
            )
            raise ValueError(msg)

        # Create directory and write labels file
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(dirname) / "labels.txt", "w") as f:
            for label in labels:
                f.write(label + "\n")

        # Export detections
        fieldnames = [
            "frame_id",
            "track_id",
            "x_min",
            "y_min",
            "w",
            "h",
            "class_id",
            "conf",
        ]
        with open(pathlib.Path(dirname) / "dets.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            for ind in range(len(self.ids)):
                item: dict[str, str | int | float] = {
                    "frame_id": self.frame_nums[ind],
                    "track_id": self.ids[ind],
                    "conf": self.confs[ind],
                    "class_id": self.classes[ind],
                    "x_min": f"{self.bboxes[ind][0]:.2f}",
                    "y_min": f"{self.bboxes[ind][1]:.2f}",
                    "w": f"{self.bboxes[ind][2]:.2f}",
                    "h": f"{self.bboxes[ind][3]:.2f}",
                }
                writer.writerow(item)

    def to_parquet(self, file_path: pathlib.Path | str) -> None:
        """
        Export detections to parquet format.

        Args:
            file_path: Relative or absolute file name to save to.

        """
        table = pa.Table.from_arrays(
            [
                pa.array(self._ids),
                pa.array(self._frame_nums),
                pa.array(self._bboxes[:, 0]),
                pa.array(self._bboxes[:, 1]),
                pa.array(self._bboxes[:, 2]),
                pa.array(self._bboxes[:, 3]),
                pa.array(self._confs),
                pa.array(self._classes),
            ],
            [
                _ID_KEY,
                _FRAME_KEY,
                _XMIN_KEY,
                _YMIN_KEY,
                _WIDTH_KEY,
                _HEIGHT_KEY,
                _CONF_KEY,
                _CLASS_KEY,
            ],
        )
        pq.write_table(table, file_path)
