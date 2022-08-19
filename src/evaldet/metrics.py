from .mot_metrics.clearmot import CLEARMOTResults, calculate_clearmot_metrics
from .mot_metrics.hota import HOTAResults, calculate_hota_metrics
from .mot_metrics.identity import IDResults, calculate_id_metrics
from .tracks import Tracks


class MOTMetricsResults(CLEARMOTResults, IDResults, HOTAResults):
    pass


def compute_mot_metrics(
    ground_truth: Tracks,
    detections: Tracks,
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
        clearmot_metrics: A sequence with the names of the metrics to compute. Allowed
            values for elements of this iterable are
        id_metrics: A sequence with the names of the metrics to compute. Allowed
            values for elements of this iterable are
        hota_metrics: A sequence with the names of the metrics to compute. Allowed
            values for elements of this iterable are

    Returns:
        A dictionary with metric names as keys, and their values as values
    """
    if not (clearmot_metrics or id_metrics or hota_metrics):
        raise ValueError("You must select some metrics to compute.")

    if not len(ground_truth):
        raise ValueError("No objects in ``ground_truths``, nothing to compute.")

    results: MOTMetricsResults = {}

    if clearmot_metrics:
        clrmt_metrics = calculate_clearmot_metrics(ground_truth, detections)
        results.update(clrmt_metrics)  # type: ignore

    if id_metrics:
        id_metrics_res = calculate_id_metrics(ground_truth, detections)
        results.update(id_metrics_res)  # type: ignore

    if hota_metrics:
        hota_metrics_res = calculate_hota_metrics(ground_truth, detections)
        results.update(hota_metrics_res)  # type: ignore

    return results
