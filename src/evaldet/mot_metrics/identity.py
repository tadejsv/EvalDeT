import typing as t

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tracks import Tracks
from .base import MOTMetricBase


class IDResults(t.TypedDict):
    IDTP: float
    IDFP: float
    IDFN: float
    IDP: float
    IDR: float
    IDF1: float


class IDMetrics(MOTMetricBase):
    def _calculate_id_metrics(
        self, ground_truth: Tracks, hypotheses: Tracks, dist_threshold: float = 0.5
    ) -> IDResults:

        gts = tuple(ground_truth.ids_count.keys())
        gts_id_ind_dict = {_id: ind for ind, _id in enumerate(gts)}
        hyps = tuple(hypotheses.ids_count.keys())
        hyps_id_ind_dict = {_id: ind for ind, _id in enumerate(hyps)}
        max_count = max(len(hyps), len(gts))

        gts_counts = np.array(tuple(ground_truth.ids_count.values()), dtype=np.int32)
        gts_counts.resize((max_count,))

        hyps_counts = np.array(tuple(hypotheses.ids_count.values()), dtype=np.int32)
        hyps_counts.resize((max_count,))

        # The "real" shape is [n_gt, n_hyp], the rest is for fictional
        # entries that are needed for FP and FN matrix to make the
        # LAP problem minimize the actual loss, including for unmatched entries
        cost_matrix = np.zeros((max_count, max_count), dtype=np.int32)
        cost_matrix += np.tile(hyps_counts, (max_count, 1))
        cost_matrix += np.tile(gts_counts, (max_count, 1)).T

        for frame in sorted(set(ground_truth.frames).intersection(hypotheses.frames)):
            dist_matrix = self._get_iou_frame(frame)
            gt_frame_inds = [gts_id_ind_dict[_id] for _id in ground_truth[frame].ids]
            htp_frame_inds = [hyps_id_ind_dict[_id] for _id in hypotheses[frame].ids]

            for gt_ind, hyp_ind in np.argwhere(dist_matrix < dist_threshold):
                cost_matrix[gt_frame_inds[gt_ind], htp_frame_inds[hyp_ind]] -= 2

        # Calculate matching as a LAP, get FN, FP and TP from matched entries
        row_m_inds, col_m_inds = linear_sum_assignment(cost_matrix)

        # Calculate the final results
        TPs = (
            gts_counts[row_m_inds].sum()
            + hyps_counts[col_m_inds].sum()
            - cost_matrix[row_m_inds, col_m_inds].sum()
        ) / 2
        FNs = gts_counts.sum() - TPs
        FPs = hyps_counts.sum() - TPs

        idp = TPs / (TPs + FPs)
        idr = TPs / (TPs + FNs)
        idf1 = 2 * TPs / (2 * TPs + FPs + FNs)

        return {
            "IDTP": TPs,
            "IDFP": FPs,
            "IDFN": FNs,
            "IDP": idp,
            "IDR": idr,
            "IDF1": idf1,
        }
