"""Module with objects for computing MOT metrics from various metric families."""

import logging
import typing as t

import numpy as np
import numpy.typing as npt

from evaldet.dist import iou_dist
from evaldet.tracks import Tracks
from evaldet.utils import timer

from .clearmot import CLEARMOTResults, calculate_clearmot_metrics
from .hota import HOTAResults, calculate_hota_metrics
from .identity import IDResults, calculate_id_metrics


class MOTMetricsResults(t.TypedDict):
    """
    A typed dictionary for storing the results of various MOT metrics evaluations.
    """

    clearmot: CLEARMOTResults | None
    id: IDResults | None
    hota: HOTAResults | None


class MOTMetrics:
    """
    The class for computing MOT metrics.

    To compute the metrics, use the ``compute`` method of this class, it will compute
    all the required MOT metrics.

    The reason for a single entrypoint for MOT computation is so that metrics can
    efficiently share pre-computed IoU distances.
    """

    _ious_dict: dict[int, npt.NDArray[np.float32]]

    def __init__(
        self, clearmot_dist_threshold: float = 0.5, id_dist_threshold: float = 0.5
    ) -> None:
        """
        Initialize the object.

        Args:
            clearmot_dist_threshold: The distance threshold for the computation of
                CLEARMOT metrics, used to determine whether a matching between two
                tracks persist, and whether to start a matching based on distance
                between two detections.
            id_dist_threshold: The distance threshold for the computation of the
                ID metrics - used to determine whether to match two objects.

        """
        self._clearmot_dist_threshold = clearmot_dist_threshold
        self._id_dist_threshold = id_dist_threshold

        self._logger = logging.getLogger(self.__class__.__name__)

    def compute(
        self,
        ground_truth: Tracks,
        hypotheses: Tracks,
        clearmot_metrics: bool = False,
        id_metrics: bool = False,
        hota_metrics: bool = True,
    ) -> MOTMetricsResults:
        """
        Compute multi-object tracking (MOT) metrics.

        Right now, the following metrics can be computed

            - CLEARMOT metrics

                - MOTA (MOT Accuracy)
                - MOTP (MOT Precision)
                - FP (false positives)
                - FN (false negatives)
                - IDS (identity switches/mismatches)

            - ID metrics

                - IDP (ID Precision)
                - IDR (ID Recall)
                - IDF1 (ID F1)
                - IDFP (ID false positives)
                - IDFN (ID false negatives)
                - IDTP (ID true positives)

            - HOTA metrics. Note that I use the matching algorithm from the paper,
            which differs from what the official repository (TrackEval) is using - see
            [this issue](https://github.com/JonathonLuiten/TrackEval/issues/22)
            for more details

                - HOTA (average and per-alpha)
                - AssA (average and per-alpha)
                - DetA (average and per-alpha)
                - LocA
                - DetPr (per-alpha)
                - DetRec (per-alpha)
                - DetTP (per-alpha)
                - DetFP (per-alpha)
                - DetFN (per-alpha)
                - AssPr (per-alpha)
                - AssRec (per-alpha)

        Args:
            ground_truth: Ground truth tracks.
            hypotheses: Hypotheses tracks.
            clearmot_metrics: Whether to compute the CLEARMOT metrics.
            id_metrics: Whether to compute the ID metrics.
            hota_metrics: Whether to compute the HOTA metrics.

        Returns:
            A dictionary of computed metrics. Metrics are saved under the key of their
            metric family (`"clearmot"`, `"id"`, `"hota"`).

        """
        if not (clearmot_metrics or id_metrics or hota_metrics):
            msg = "You must select some metrics to compute."
            raise ValueError(msg)

        if not len(ground_truth):
            msg = "No objects in ``ground_truths``, nothing to compute."
            raise ValueError(msg)

        with timer.timer(self._logger, "Precompute IoU"):
            self._ious_dict = _compute_ious(ground_truth, hypotheses)

        if clearmot_metrics:
            with timer.timer(self._logger, "Compute CLEARMOT Metrics"):
                clrmt_metrics = calculate_clearmot_metrics(
                    ground_truth,
                    hypotheses,
                    self._ious_dict,
                    self._clearmot_dist_threshold,
                )
        else:
            clrmt_metrics = None

        if id_metrics:
            with timer.timer(self._logger, "Compute ID Metrics"):
                id_metrics_res = calculate_id_metrics(
                    ground_truth, hypotheses, self._ious_dict, self._id_dist_threshold
                )
        else:
            id_metrics_res = None

        if hota_metrics:
            with timer.timer(self._logger, "Compute HOTA Metrics"):
                hota_metrics_res = calculate_hota_metrics(
                    ground_truth,
                    hypotheses,
                    self._ious_dict,
                )
        else:
            hota_metrics_res = None

        # Remove IoU matrix to release memory
        del self._ious_dict

        return MOTMetricsResults(
            clearmot=clrmt_metrics, id=id_metrics_res, hota=hota_metrics_res
        )


def _compute_ious(
    tracks_1: Tracks, tracks_2: Tracks
) -> dict[int, npt.NDArray[np.float32]]:
    all_frames = sorted(set(tracks_1.frames).intersection(tracks_2.frames))

    ious: dict[int, npt.NDArray[np.float32]] = {}
    for frame in all_frames:
        ious_f = iou_dist(tracks_1[frame].bboxes, tracks_2[frame].bboxes)
        ious[frame] = ious_f

    return ious
