from typing import Optional

import numpy as np
import numpy.typing as npt


def prec_recall_curve(
    hyp_matched: npt.NDArray[np.bool_],
    hyp_conf: npt.NDArray[np.float32],
    n_gts: int,
) -> Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Calculate the precision-recall curve.

    This calculates the precision-recall curve for matched hypotheses and ground truths.
    There are two possible edge cases:
    * when there are no ground truths: in this case `None` is returned
    * when there are no hypotheses (predictions): in this case the PR curve consists
      of a single `(0,0)` point.

    Args:
        hyp_matched: A `[N,]` array indicating whether a hypotheses is matched or not.
        hyp_conf: A `[N,]` array of hypotheses' confidence scores.
        n_gts: Number of ground truths

    Returns:
        If `n_gts == 0`, the function returns `None`. Otherwise two `L` length arrays
        are returned: `precision` and `recall`. The contain all points on the precision
        recall curve.

        If there are no hypotheses (`N==0`), by convention the precision-recall curve
        will contain only the `(0,0)` point.
    """

    if n_gts == 0:
        return None

    if len(hyp_matched) == 0:
        return (np.array([0.0]), np.array([0]))

    sort_arr = np.argsort(-hyp_conf)
    hyp_conf = hyp_conf[sort_arr]
    hyp_matched = hyp_matched[sort_arr]

    precision = np.cumsum(hyp_matched) / np.arange(1, len(hyp_matched) + 1)
    recall = np.cumsum(hyp_matched) / n_gts

    return precision, recall
