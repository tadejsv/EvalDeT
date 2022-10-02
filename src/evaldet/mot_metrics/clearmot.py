import logging
import typing as t

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tracks import Tracks
from .base import MOTMetricBase

logger = logging.getLogger(__name__)


class CLEARMOTResults(t.TypedDict):
    MOTP: float
    MOTA: float
    FP_CLEAR: float
    FN_CLEAR: float
    IDSW: float


class CLEARMOTMetrics(MOTMetricBase):
    def _calculate_clearmot_metrics(
        self, ground_truth: Tracks, hypotheses: Tracks, dist_threshold: float = 0.5
    ) -> CLEARMOTResults:

        all_frames = sorted(ground_truth.frames.union(hypotheses.frames))

        false_negatives = 0
        false_positives = 0
        mismatches = 0
        ground_truths = 0

        matches_dist = []
        matching: t.Dict[int, int] = {}

        # This is the persistent matching dictionary, used to check for mismatches
        # when a previously matched hypothesis is re-matched with a ground truth
        matching_persist: t.Dict[int, int] = {}

        for frame in all_frames:
            if frame not in ground_truth:
                matching = {}
                false_positives += len(hypotheses[frame].ids)
            elif frame not in hypotheses:
                matching = {}
                false_negatives += len(ground_truth[frame].ids)
                ground_truths += len(ground_truth[frame].ids)
            else:
                hyp, gt = hypotheses[frame], ground_truth[frame]
                ground_truths += len(gt.ids)

                # Delete from matching missing gts/detections
                for key in list(matching.keys()):
                    if matching[key] not in hyp.ids or key not in gt.ids:
                        del matching[key]

                ious = self._get_iou_frame(frame)
                id_ind_dict_gt = {_id: ind for ind, _id in enumerate(gt.ids)}
                id_ind_dict_hyp = {_id: ind for ind, _id in enumerate(hyp.ids)}

                for gt_id, hyp_id in tuple(matching.items()):
                    dist_match = ious[id_ind_dict_gt[gt_id], id_ind_dict_hyp[hyp_id]]
                    if dist_match > dist_threshold:
                        del matching[gt_id]
                    else:
                        matches_dist.append(dist_match)

                # For remaining (non-matching) gts/detections, compute matching as a LAP
                match_gt_ids = tuple(matching.keys())
                match_hyp_ids = tuple(matching.values())
                inds_nm_gt = np.where(np.isin(gt.ids, match_gt_ids, invert=True))[0]
                inds_nm_hyp = np.where(np.isin(hyp.ids, match_hyp_ids, invert=True))[0]
                dist_matrix = ious[np.ix_(inds_nm_gt, inds_nm_hyp)]

                for row_ind, col_ind in zip(*linear_sum_assignment(dist_matrix)):
                    if dist_matrix[row_ind, col_ind] < dist_threshold:
                        matching[gt.ids[inds_nm_gt[row_ind]]] = hyp.ids[
                            inds_nm_hyp[col_ind]
                        ]
                        matches_dist.append(dist_matrix[row_ind, col_ind])

                # Check if mismatches have occured
                for key in matching:
                    if (
                        key in matching_persist
                        and matching[key] != matching_persist[key]
                    ):
                        mismatches += 1

                # Compute false positives and false negatices
                false_negatives += len(gt.ids) - len(matching)
                false_positives += len(hyp.ids) - len(matching)
                matching_persist.update(matching)

        if not matches_dist:
            logger.warning("No matches were made, MOPT will be np.nan")
            motp = np.nan
        else:
            motp = sum(matches_dist) / len(matches_dist)

        mota = 1 - (false_negatives + false_positives + mismatches) / ground_truths

        return {
            "MOTP": motp,
            "MOTA": mota,
            "FP_CLEAR": false_positives,
            "FN_CLEAR": false_negatives,
            "IDSW": mismatches,
        }
