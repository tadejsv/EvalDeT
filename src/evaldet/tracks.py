import collections as co
import csv
import datetime as dt
import pathlib
import typing as t
import xml.etree.ElementTree as ET

import numpy as np

_ID_KEY = "id"
_FRAME_KEY = "frame"
_CONF_KEY = "conf"
_CLASS_KEY = "class"
_XMIN_KEY = "xmin"
_YMIN_KEY = "ymin"
_WIDTH_KEY = "width"
_HEIGHT_KEY = "height"


class FrameTracks(t.NamedTuple):
    detections: np.ndarray
    ids: np.ndarray
    classes: np.ndarray
    confs: np.ndarray


TracksType = t.TypeVar("TracksType", bound="Tracks")


class Tracks:
    """A class representing objects' tracks in a MOT setting.

    It allows for the tracks to be manually constructed, frame by frame,
    but also provides convenience class methods to initialize it from
    a file, the following formats are currently supported

    - MOT format (as described `here <https://motchallenge.net/instructions/>`__)
    - MOT ground truth format (as described `here <https://arxiv.org/abs/1603.00831>`__)
    - CVAT's version of the MOT format (as described `here <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/formats/format-mot/>`__)
    - CVAT for Video format (as described `here <https://openvinotoolkit.github.io/cvat/docs/manual/advanced/xml_format/>`__)
    - UA-DETRAC XML format (you can download an example `here <https://detrac-db.rit.albany.edu/Tracking>`__)

    The frame numbers will be zero-indexed internally, so for the MOT files 1 will be
    subtracted from all frame numbers.
    """

    @classmethod
    def from_csv(
        cls: t.Type[TracksType],
        csv_file: t.Union[str, pathlib.Path],
        fieldnames: t.List[str],
        zero_indexed: bool = True,
    ) -> TracksType:
        """Get detections from a CSV file.

        The CSV file should have a normal comma (,) as a separator, and should not
        include a header.

        Args:
            csv_file: path to the CSV file
            filednames: The names of the fields. This will be passed to
                ``csv.DictReader``. It should contain the names of the fields, in order
                that they appear. The following names will be used (others will be
                disregarded):
                - ``xmin``
                - ``ymin``
                - ``height``
                - ``width``
                - ``conf``: for the confidence of the item
                - ``class``: for the class label of the item
                - ``id``: for the id of the item
                - ``frame``: for the frame number
            zero_indexed: If the frame numbers are zero indexed. Otherwise they are
                assumed to be 1 indexed, and 1 will be subtracted from all frame numbers
                to make them zero indexed.
        """
        tracks = cls()
        with open(csv_file, newline="") as file:
            csv_reader = csv.DictReader(file, fieldnames=fieldnames, dialect="unix")
            frames: t.Dict[int, t.Any] = {}

            for line_num, line in enumerate(csv_reader):
                try:
                    cls._add_to_tracks_accumulator(frames, line)

                except ValueError as e:
                    raise ValueError(
                        "Error when converting values to numbers on line"
                        f" {line_num}. Please check that all the values are numeric"
                        " and that the file follows the MOT format."
                    ) from e

            list_frames = sorted(list(frames.keys()))
            for frame in list_frames:
                extra_vals = {}
                if _CONF_KEY in fieldnames:
                    extra_vals["confs"] = frames[frame]["confs"]
                if _CLASS_KEY in fieldnames:
                    extra_vals["classes"] = frames[frame]["classes"]

                if zero_indexed:
                    frame_num = frame
                else:
                    frame_num = frame - 1

                tracks.add_frame(
                    frame_num,
                    ids=frames[frame]["ids"],
                    detections=np.array(frames[frame]["detections"]),
                    **extra_vals,
                )

        return tracks

    @classmethod
    def from_mot(
        cls: t.Type[TracksType], file_path: t.Union[pathlib.Path, str]
    ) -> TracksType:
        """Creates a Tracks object from detections file in the MOT format.

        The format should look like this::

            <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <conf>, <x>, <y>, <z>

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The values for ``x``, ``y`` and ``z`` will be ignored.

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
    def from_mot_gt(
        cls: t.Type[TracksType], file_path: t.Union[pathlib.Path, str]
    ) -> TracksType:
        """Creates a Tracks object from detections file in the MOT ground truth format.
        This format has some more information compared to the normal

        The format should look like this::

            <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <conf>, <class>, <visibility>

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The value for ``visibility`` will be ignored.

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
    def from_mot_cvat(
        cls: t.Type[TracksType], file_path: t.Union[pathlib.Path, str]
    ) -> "Tracks":
        """Creates a Tracks object from detections file in the CVAT's MOT format.

        The format should look like this::

            <frame>, <id>, <xmin>, <ymin>, <width>, <height>, <not ignored>, <class>, <visibility>, <skipped>

        Note that all values above are expected to be **numeric** - string values will
        cause an error. The last two elements (``visibility`` and ``skipped``) are
        optional. The values for ``not ignored``, ``visibility`` and ``skipped`` will be
        ignored.

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
            "_",
            _CLASS_KEY,
        ]

        return cls.from_csv(file_path, fieldnames, zero_indexed=False)

    @classmethod
    def from_ua_detrac(
        cls: t.Type[TracksType],
        file_path: t.Union[pathlib.Path, str],
        classes_attr_name: t.Optional[str] = None,
        classes_list: t.Optional[t.List[str]] = None,
    ) -> TracksType:
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

        frames = root.findall(_FRAME_KEY)
        for frame in frames:
            tracks_f = frame.find("target_list").findall("target")  # type: ignore

            current_frame = int(frame.attrib["num"])
            detections: t.List[t.List[float]] = []
            classes: t.List[int] = []
            ids: t.List[int] = []

            for track in tracks_f:
                # Get track attributes
                ids.append(int(track.attrib[_ID_KEY]))

                box = track.find("box")
                xmin, ymin = float(box.attrib["left"]), float(box.attrib["top"])  # type: ignore
                width = float(box.attrib[_WIDTH_KEY])  # type: ignore
                height = float(box.attrib[_HEIGHT_KEY])  # type: ignore
                detections.append([xmin, ymin, width, height])

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
        cls: t.Type[TracksType],
        file_path: t.Union[pathlib.Path, str],
        classes_list: t.List[str],
    ) -> TracksType:
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

        Elements with "outside=1" will be ignored.

        Args:
            file_path: Path where the detections file is located
            classes_list: The list of all possible class values. The values from that
                attribute in the file will then be replaced by the index of that value
                in this list.
        """

        xml_tree = ET.parse(file_path)
        root = xml_tree.getroot()
        tracks = cls()

        frames: t.Dict[int, t.Dict[str, t.List]] = co.defaultdict(
            lambda: co.defaultdict(list)
        )
        tracks_cvat = root.findall("track")
        for track_cvat in tracks_cvat:
            track_id = int(track_cvat.attrib[_ID_KEY])
            track_class = classes_list.index(track_cvat.attrib["label"])

            for box in track_cvat.findall("box"):
                if int(box.attrib["outside"]) == 1:
                    continue

                frame_num = int(box.attrib[_FRAME_KEY])
                xmin, ymin = float(box.attrib["xtl"]), float(box.attrib["ytl"])
                width = float(box.attrib["xbr"]) - xmin
                height = float(box.attrib["ybr"]) - ymin

                current_frame = frames[frame_num]
                current_frame["classes"].append(track_class)
                current_frame["ids"].append(track_id)
                current_frame["detections"].append([xmin, ymin, width, height])

        list_frames = sorted(list(frames.keys()))
        for frame in list_frames:
            tracks.add_frame(
                frame,
                ids=frames[frame]["ids"],
                detections=np.array(frames[frame]["detections"]),
                classes=frames[frame]["classes"],
            )

        return tracks

    @staticmethod
    def _add_to_tracks_accumulator(frames: dict, new_obj: dict) -> None:
        frame_num, track_id = int(new_obj[_FRAME_KEY]), int(float(new_obj[_ID_KEY]))

        xmin, ymin = float(new_obj[_XMIN_KEY]), float(new_obj[_YMIN_KEY])
        width, height = float(new_obj[_WIDTH_KEY]), float(new_obj[_HEIGHT_KEY])

        if _CONF_KEY in new_obj:
            conf = float(new_obj[_CONF_KEY])
        if _CLASS_KEY in new_obj:
            class_id = int(new_obj[_CLASS_KEY])

        current_frame = frames.get(frame_num, {})

        # If some detections already exist for the current frame
        if current_frame:
            if _CONF_KEY in new_obj:
                current_frame["confs"].append(conf)
            if _CLASS_KEY in new_obj:
                current_frame["classes"].append(class_id)

            current_frame["ids"].append(track_id)
            current_frame["detections"].append([xmin, ymin, width, height])
        else:
            if _CONF_KEY in new_obj:
                current_frame["confs"] = [conf]
            if _CLASS_KEY in new_obj:
                current_frame["classes"] = [class_id]

            current_frame["ids"] = [track_id]
            current_frame["detections"] = [[xmin, ymin, width, height]]
            frames[frame_num] = current_frame

    def __init__(self) -> None:

        self._frames: t.Set[int] = set()
        self._detections: t.Dict[int, np.ndarray] = dict()
        self._ids: t.Dict[int, np.ndarray] = dict()
        self._classes: t.Dict[int, np.ndarray] = dict()
        self._confs: t.Dict[int, np.ndarray] = dict()
        self._id_to_frames: t.Dict[int, t.Set[int]] = co.defaultdict(set)

    def add_frame(
        self,
        frame_num: int,
        ids: t.Union[t.List[int], np.ndarray],
        detections: np.ndarray,
        classes: t.Optional[t.Union[t.List[int], np.ndarray]] = None,
        confs: t.Optional[t.Union[t.List[float], np.ndarray]] = None,
    ) -> None:
        """Add a frame to the collection. Can overwrite an existing frame.

        Args:
            frame_num: A non-negative frame number
            ids: A list or a numpy array with ids of the objects in the frame.
            detections: An Nx4 array describing the bounding boxes of objects in
                the frame. It should be in the ``xywh`` format.
            classes: An optional list (or numpy array) of classes for the objects.
                If not passed all objects in the frame will get a class of 0.
            confs: An optional list (or numpy array) of confidence scores for the
                objects. If not passed, all detection will get a confidence of 1.
        """

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

        if confs is not None and len(confs) != len(ids):
            raise ValueError(
                "If `confs` is given, it should contain the same number of items"
                " as `ids`."
            )

        if detections.shape[1] != 4:
            raise ValueError(
                "The `detections` should be an Nx4 array, but got"
                f" shape Nx{detections.shape[1]}"
            )

        if len(set(ids)) != len(ids):
            raise ValueError(
                f"The `ids` must be unique - got non-unique ids on frame {frame_num}"
            )

        # If all ok, add objects to collections
        self._detections[frame_num] = np.array(detections, copy=True, dtype=np.float32)
        self._ids[frame_num] = np.array(ids, copy=True, dtype=np.int32)

        if classes is not None:
            self._classes[frame_num] = np.array(classes, copy=True, dtype=np.int32)
        else:
            self._classes[frame_num] = np.full((len(ids),), 0, dtype=np.int32)

        if confs is not None:
            self._confs[frame_num] = np.array(confs, copy=True, dtype=np.float32)
        else:
            self._confs[frame_num] = np.full((len(ids),), 1, dtype=np.float32)

        self._frames.add(frame_num)

        for _id in ids:
            self._id_to_frames[_id].add(frame_num)

    def filter_frame(self, frame_num: int, filter: np.ndarray) -> None:
        """Filters a frame and all its objects according to the filter array.

        If the result would be that all objects in the frame are filtered out,
        the frame is deleted.

        Args:
            frame_num: number of the frame to filter
            filter: A boolean array indicating which entries to keep
                (where it equals ``True``). Should have the same shape as ids
                of the frame it is trying to filter.
        """
        frame = self[frame_num]

        if filter.shape != frame.ids.shape:
            raise ValueError(
                f"Filter must be the same shape as ids, {frame.ids.shape},"
                f" but got {filter.shape}"
            )

        if filter.dtype != bool:
            raise ValueError(
                f"Filter must be a boolean array, instead got type {filter.dtype}"
            )

        if filter.sum() == 0:
            del self[frame_num]
            return

        if filter.min() == 1:
            return

        for id_out in frame.ids[~filter]:
            if len(self._id_to_frames[id_out]) == 1:
                del self._id_to_frames[id_out]
            else:
                self._id_to_frames[id_out].remove(frame_num)

        self._detections[frame_num] = frame.detections[filter]
        self._ids[frame_num] = frame.ids[filter]
        self._confs[frame_num] = frame.confs[filter]
        self._classes[frame_num] = frame.classes[filter]

    def filter_by_class(self, classes: t.List[int]) -> None:
        """Filter all frames by classes

        This will keep the detections with class label corresponding to one of the
        classes passed in ``classes``. If a frame does not have class labels, or
        would have filtered out all the items, it is deleted.

        Args:
            classes: A list of which class labels to keep.
        """
        for frame in self._frames.copy():
            filter_cls = np.in1d(self._classes[frame], classes)
            self.filter_frame(frame, filter_cls)

    def filter_by_conf(self, lower_bound: float) -> None:
        """Filter all frames by confidence

        This will keep the detections with confidence value higher or equal to
        the ``lower_bound``. If a frame does not have confidence labels, or
        would have filtered out all the items, it is deleted.

        Args:
            lower_bound: The lower bound on the confidence value of the items,
                so that they are not filtered out.
        """
        for frame in self._frames.copy():
            filter_conf = self._confs[frame] >= lower_bound
            self.filter_frame(frame, filter_conf)

    @property
    def all_classes(self) -> t.Set[int]:
        """Get a set of all classes in the collection."""
        classes: t.Set[int] = set()
        for frame in self._frames:
            if frame in self._classes:
                classes.update(self._classes[frame])

        return classes

    @property
    def ids_count(self) -> t.Dict[int, int]:
        """Get the number of frames that each id is present in.

        Returns:
            A dictionary where keys are the track ids, and values
            are the numbers of frames they appear in.
        """
        ids_count = {k: len(v) for k, v in self._id_to_frames.items()}
        return ids_count

    @property
    def frames(self) -> t.Set[int]:
        """Get an ordered list of all frame numbers in the collection."""
        return self._frames.copy()

    @property
    def id_to_frames(self) -> t.Dict[int, t.Set[int]]:
        """Get an ordered list of all frame numbers in the collection."""
        return {k: v.copy() for k, v in self._id_to_frames.items()}

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, idx: int) -> bool:
        """Whether the frame ``idx`` is present in the collection."""
        return idx in self._frames

    @t.overload
    def __getitem__(self, idx: int) -> FrameTracks:
        """Get the frame with number ``idx``.

        Note that indexing with negative values is not supported.
        """

    @t.overload
    def __getitem__(self, idx: slice) -> "Tracks":
        """Select only a subset of frames, as defined by the slice.

        Note that the ``step`` argument is not supported and will result in an error
        being raised if it is supplied. Negative indices for start or stop argument are
        similarly not supported.
        """

    def __getitem__(self, idx: t.Union[int, slice]) -> t.Union[FrameTracks, "Tracks"]:

        if isinstance(idx, int):
            if idx < 0:
                raise ValueError("Indexing with negative values is not supported.")
            if idx not in self:
                raise KeyError(f"The frame {idx} does not exist.")

            return FrameTracks(
                ids=self._ids[idx],
                detections=self._detections[idx],
                classes=self._classes[idx],
                confs=self._confs[idx],
            )

        elif isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if step is not None:
                raise ValueError("Slicing with the step argument is not supported.")
            if start < 0 or stop < 0:
                raise ValueError("Slicing with negative indices is not supported.")

            keep_frames = set(list(range(start, stop)))
            keep_frames = self._frames.intersection(keep_frames)

            new_tracks = type(self)()

            new_tracks._frames = keep_frames
            new_tracks._detections.update(
                {fnum: self._detections[fnum].copy() for fnum in keep_frames}
            )
            new_tracks._ids.update(
                {fnum: self._ids[fnum].copy() for fnum in keep_frames}
            )
            new_tracks._classes.update(
                {fnum: self._classes[fnum].copy() for fnum in keep_frames}
            )
            new_tracks._confs.update(
                {fnum: self._confs[fnum].copy() for fnum in keep_frames}
            )

            new_id_to_frames = {
                k: v.intersection(keep_frames) for k, v in self._id_to_frames.items()
            }
            new_tracks._id_to_frames.update(
                {k: v for k, v in new_id_to_frames.items() if len(v) > 0}
            )
            return new_tracks
        else:
            raise ValueError("Unrecognized index type")

    def __delitem__(self, frame_num: int) -> None:
        """Remove the frame with number ``frame_num``"""

        if frame_num not in self:
            raise KeyError(f"Tracks object does not contain frame {frame_num}.")

        for id_out in self._ids[frame_num]:
            if len(self._id_to_frames[id_out]) == 1:
                del self._id_to_frames[id_out]
            else:
                self._id_to_frames[id_out].remove(frame_num)

        self._frames.remove(frame_num)
        del self._ids[frame_num]
        del self._detections[frame_num]
        del self._classes[frame_num]
        del self._confs[frame_num]

    def to_cvat_video(
        self,
        filename: t.Union[pathlib.Path, str],
        labels: t.Sequence[str],
        image_size: t.Tuple[int, int] = (1, 1),
    ) -> None:
        """Export detections to CVAT for Video 1.1 format.

        More information on the format can be found `here <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_.

        Args:
            filename: The name of the file to save to - should have an ``.xml`` suffix.
            labels: A list/tuple of label names. The length should be at least the
                maximum label index - 1 (the first label corresponds to label at the
                0-th index).
            image_size: The size of the image in the ``[w, h]`` format, in pixels.
        """

        if len(self._id_to_frames) == 0:
            max_label_ind = 1
        else:
            max_label_ind = max(self.all_classes) + 1
        if len(labels) < max_label_ind:
            raise ValueError(
                f"The length of provied labels {len(labels)} is less than the largest"
                f" label id (+1) among tracklets {max_label_ind}."
            )

        w, h = image_size
        if len(self._frames) == 0:
            max_frame = 1
        else:
            max_frame = max(self._frames)

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

        for track_id, track_frames_set in self._id_to_frames.items():
            track_frames = sorted(list(track_frames_set))

            # Take first label, as only one is supported
            id_first_ind = np.nonzero(self._ids[track_frames[0]] == track_id)[0][0]
            label: int = self._classes[track_frames[0]][id_first_ind]

            track_el = ET.SubElement(
                annotations,
                "track",
                id=str(track_id),
                label=labels[label],
                source=type(self).__name__,
            )

            for i, frame_num in enumerate(track_frames):
                id_frame_ind = np.nonzero(self._ids[frame_num] == track_id)[0][0]
                bbox = self._detections[frame_num][id_frame_ind]
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
                if (i + 1) == len(track_frames) or (frame_num + 1) < track_frames[
                    i + 1
                ]:
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
        dirname: t.Union[pathlib.Path, str],
        labels: t.Sequence[str],
    ) -> None:
        """Export detections to a simple CSV format. The format comprises of two files:
        ``dets.csv``, containing the detections, and ``labels.txt``, which contains
        the names of the labels (corresponding to label indices in ``dets.csv``). The
        rows in ``dets.csv`` have the following format:

            <frame>, <id>, <x_min>, <y_min>, <width>, <height>, <class>, <conf>

        Note that ``frame`` and ``class`` are both 0 indexed.

        Args:
            dirname: The name of the directory to save to - will be created if it
                doesn't already exist.
            labels: A list/tuple of label names. The length should be at least the
                maximum label index - 1 (the first label corresponds to label at the
                0-th index).
        """

        if len(self._id_to_frames) == 0:
            max_label_ind = 1
        else:
            max_label_ind = max(self.all_classes) + 1
        if len(labels) < max_label_ind:
            raise ValueError(
                f"The length of provied labels {len(labels)} is less than the largest"
                f" label id (+1) among tracklets {max_label_ind}."
            )

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

            for frame in sorted(self._frames):
                dets = self._detections[frame]
                ids = self._ids[frame]
                classes = self._classes[frame]
                confs = self._confs[frame]

                for det_idx in range(len(dets)):
                    item: t.Dict[str, t.Union[str, int, float]] = {
                        "frame_id": frame,
                        "track_id": ids[det_idx],
                        "conf": confs[det_idx],
                        "class_id": classes[det_idx],
                        "x_min": f"{dets[det_idx][0]:.2f}",
                        "y_min": f"{dets[det_idx][1]:.2f}",
                        "w": f"{dets[det_idx][2]:.2f}",
                        "h": f"{dets[det_idx][3]:.2f}",
                    }
                    writer.writerow(item)
