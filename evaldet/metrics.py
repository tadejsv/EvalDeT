from typing import Dict, Sequence, Union

from .mot_metrics.clearmot import calculate_clearmot_metrics
from .tracks import Tracks

_CLEARMOT_METRICS = ("mota", "motp")
_ID_METRICS = ("idp", "idr", "idf1")
_ALLOWED_MOT_METRICS = _CLEARMOT_METRICS + _ID_METRICS


def compute_mot_metrics(
    metrics: Sequence[str], ground_truth: Tracks, detections: Tracks
) -> Dict[str, Union[int, float]]:
    """Compute multi-object tracking (MOT) metrics.

    Right now, the following metrics can be computed:
        - CLEARMOT metrics:
            - MOTA (MOT Accuracy)
            - MOTP (MOT Precision)
        - ID metrics:
            - IDP (ID Precision)
            - IDR (ID Recall)
            - IDF1 (ID F1)

    Args:
        metrics: A sequence with the names of the metrics to compute. Allowed
            values for elements of this iterable are:
            - ``'mota'``: MOTA metric
            - ``'motp'``: MOTP metric
            - ``'idp'``: ID Precision metric
            - ``'idr'``: ID Recall metric
            - ``'idf1'``: ID F1 metric
        ground_truth: A :class:`evaldet.tracks.Tracks` object, representing ground
            truth annotations.
        detections: A :class:`evaldet.tracks.Tracks` object, representing detection
            annotations ().

    Returns:
        metrics_results: A dictionary with metric names as keys, and their values as
            values
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

    results = {}

    mot_metrics = set(_CLEARMOT_METRICS).intersection(metrics)
    if mot_metrics:
        mot_metrics_results = calculate_clearmot_metrics(ground_truth, detections)
        for metric_name in mot_metrics:
            results[metric_name] = mot_metrics_results[metric_name]

    return results
