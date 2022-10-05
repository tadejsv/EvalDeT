import collections as co
import typing as t

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tracks import Tracks
from ..utils import sparse
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

        gts_counts = np.array(tuple(ground_truth.ids_count.values()), dtype=np.int32)
        hyps_counts = np.array(tuple(hypotheses.ids_count.values()), dtype=np.int32)

        matches: t.Dict[t.Tuple[int, int], int] = co.defaultdict(int)
        for frame in sorted(set(ground_truth.frames).intersection(hypotheses.frames)):
            dist_matrix = self._get_iou_frame(frame)
            gt_frame_inds = [gts_id_ind_dict[_id] for _id in ground_truth[frame].ids]
            htp_frame_inds = [hyps_id_ind_dict[_id] for _id in hypotheses[frame].ids]

            for gt_ind, hyp_ind in np.argwhere(dist_matrix < dist_threshold):
                matches[(gt_frame_inds[gt_ind], htp_frame_inds[hyp_ind])] += 1

        # Calculate matching as a LAP, get FN, FP and TP from matched entries
        # row_m_inds, col_m_inds = linear_sum_assignment(cost_matrix)
        matches_matrix = sparse.create_coo_array(matches, (len(gts), len(hyps)))
        matches_array = matches_matrix.toarray()
        row_m_inds, col_m_inds = linear_sum_assignment(matches_array, maximize=True)

        # Calculate the final results
        TPs = matches_array[row_m_inds, col_m_inds].sum()
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
