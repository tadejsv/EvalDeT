from typing import Dict, Sequence, Union

import numpy as np

from .mot_metrics.clearmot import calculate_clearmot_metrics
from .mot_metrics.hota import calculate_hota_metrics
from .mot_metrics.identity import calculate_id_metrics
from .tracks import Tracks

_CLEARMOT_METRICS = ("MOTA", "MOTP", "FN_CLEAR", "FP_CLEAR", "IDS")
_ID_METRICS = ("IDP", "IDR", "IDF1", "IDFP", "IDFN", "IDTP")
_HOTA_METRICS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "alphas_HOTA",
    "HOTA_alpha",
    "DetA_alpha",
    "AssA_alpha",
    "LocA_alpha",
)
_ALLOWED_MOT_METRICS = _CLEARMOT_METRICS + _ID_METRICS + _HOTA_METRICS


def compute_mot_metrics(
    metrics: Sequence[str], ground_truth: Tracks, detections: Tracks
) -> Dict[str, Union[int, float, np.ndarray]]:
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
        metrics: A sequence with the names of the metrics to compute. Allowed
            values for elements of this iterable are

            - ``'MOTA'``: MOTA metric
            - ``'MOTP'``: MOTP metric
            - ``'FP_CLEAR'``: False positive detections (from CLEARMOT)
            - ``'FN_CLEAR'``: False negative detections (from CLEARMOT)
            - ``'IDS'``: Identity switches/mismatches (from CLEARMOT)
            - ``'IDP'``: ID Precision metric
            - ``'IDR'``: ID Recall metric
            - ``'IDF1'``: ID F1 metric
            - ``'IDFP'``: ID false positives
            - ``'IDFN'``: ID false negatives
            - ``'IDTP'``: ID true positives
            - ``'HOTA'``: The HOTA metric
            - ``'AssA'``: The AssA metric (from HOTA)
            - ``'DetA'``: The DetA metric (from HOTA)
            - ``'LocA'``: The LocA metric (from HOTA)
            - ``'alphas_HOTA'``: The alpha values used in HOTA calculation
            - ``'HOTA_alpha'``: An array of HOTA values for each alpha
            - ``'AssA_alpha'``: An array of AssA values for each alpha
            - ``'DetA_alpha'``: An array of DetA values for each alpha
            - ``'LocA_alpha'``: An array of LocA values for each alpha

        ground_truth: A :class:`evaldet.tracks.Tracks` object, representing ground
            truth annotations.
        detections: A :class:`evaldet.tracks.Tracks` object, representing detection
            annotations ().

    Returns:
        A dictionary with metric names as keys, and their values as values
    """

    if not set(metrics).issubset(_ALLOWED_MOT_METRICS):
        extra_metrics = set(metrics) - set(_ALLOWED_MOT_METRICS)
        raise ValueError(
            "These elements in ``metrics`` are not among allowed metrics:"
            f" {list(extra_metrics)}"
        )

    if not len(metrics):
        raise ValueError("The ``metrics`` sequence is empty, nothing to compute.")

    if not len(ground_truth):
        raise ValueError("No objects in ``ground_truths``, nothing to compute.")

    results: Dict[str, Union[int, float, np.ndarray]] = {}

    mot_metrics = set(_CLEARMOT_METRICS).intersection(metrics)
    if mot_metrics:
        clearmot_metrics_results = calculate_clearmot_metrics(ground_truth, detections)
        for metric_name in mot_metrics:
            results[metric_name] = clearmot_metrics_results[metric_name]

    id_metrics = set(_ID_METRICS).intersection(metrics)
    if id_metrics:
        id_metrics_results = calculate_id_metrics(ground_truth, detections)
        for metric_name in id_metrics:
            results[metric_name] = id_metrics_results[metric_name]

    hota_metrics = set(_HOTA_METRICS).intersection(metrics)
    if hota_metrics:
        hota_metrics_results = calculate_hota_metrics(ground_truth, detections)
        for metric_name in hota_metrics:
            results[metric_name] = hota_metrics_results[metric_name]

    return results
