# import logging
# import time
import typing as t

import numpy as np

from .dist import iou_dist
from .mot_metrics.clearmot import CLEARMOTMetrics, CLEARMOTResults
from .mot_metrics.hota import HOTAMetrics, HOTAResults
from .mot_metrics.identity import IDMetrics, IDResults
from .tracks import Tracks


class MOTMetricsResults(t.TypedDict):
    clearmot: t.Optional[CLEARMOTResults]
    id: t.Optional[IDResults]
    hota: t.Optional[HOTAResults]


class MOTMetrics(CLEARMOTMetrics, IDMetrics, HOTAMetrics):
    def __init__(
        self, clearmot_dist_threshold: float = 0.5, id_dist_threshold: float = 0.5
    ) -> None:
        self._clearmot_dist_threshold = clearmot_dist_threshold
        self._id_dist_threshold = id_dist_threshold

        self._ious: t.Optional[list[np.ndarray]] = None
        self._ious_dict: t.Optional[dict[int, int]] = None

    def _precompute_iou(self, ground_truth: Tracks, hypotheses: Tracks) -> None:
        all_frames = sorted(set(ground_truth.frames).intersection(hypotheses.frames))

        self._ious = []
        self._ious_dict = {}
        for ind, frame in enumerate(all_frames):
            ious_f = iou_dist(
                ground_truth[frame].detections, hypotheses[frame].detections
            )
            self._ious.append(ious_f)
            self._ious_dict[frame] = ind

    def _get_iou_frame(self, frame: int) -> np.ndarray:
        # Keeps mypy happy
        assert self._ious is not None and self._ious_dict is not None

        return self._ious[self._ious_dict[frame]]

    def compute(
        self,
        ground_truth: Tracks,
        hypotheses: Tracks,
        clearmot_metrics: bool = False,
        id_metrics: bool = False,
        hota_metrics: bool = True,
    ) -> MOTMetricsResults:
        """Compute multi-object tracking (MOT) metrics.

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

            - HOTA metrics (both average and individual alpha values). Note that I use
            the matching algorithm from the paper, which differs from what the official
            repository (TrackEval) is using - see
            `this issue <https://github.com/JonathonLuiten/TrackEval/issues/22>`__
            for more details

                - HOTA
                - AssA
                - DetA
                - LocA

        Args:
            ground_truth: Ground truth tracks.
            hypotheses: Hypotheses tracks.
            clearmot_metrics: Whether to compute the CLEARMOT metrics.
            id_metrics: Whether to compute the ID metrics.
            hota_metrics: Whether to compute the HOTA metrics.

        Returns:
            A dictionary with metric names as keys, and their values as values
        """
        if not (clearmot_metrics or id_metrics or hota_metrics):
            raise ValueError("You must select some metrics to compute.")

        if not len(ground_truth):
            raise ValueError("No objects in ``ground_truths``, nothing to compute.")

        self._precompute_iou(ground_truth, hypotheses)

        if clearmot_metrics:
            clrmt_metrics = self._calculate_clearmot_metrics(
                ground_truth, hypotheses, self._clearmot_dist_threshold
            )
        else:
            clrmt_metrics = None

        if id_metrics:
            id_metrics_res = self._calculate_id_metrics(
                ground_truth, hypotheses, self._id_dist_threshold
            )
        else:
            id_metrics_res = None

        if hota_metrics:
            hota_metrics_res = self._calculate_hota_metrics(ground_truth, hypotheses)
        else:
            hota_metrics_res = None

        # Remove IoU matrix to release memory
        self._ious = None
        self._ious_dict = None

        return MOTMetricsResults(
            clearmot=clrmt_metrics, id=id_metrics_res, hota=hota_metrics_res
        )
