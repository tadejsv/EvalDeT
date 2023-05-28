import numba as nb
import numpy as np
import numpy.typing as npt
from numba import types

from evaldet import Detections
from evaldet.dist import iou_dist


class DetMetricBase:
    @staticmethod
    def _match_images(
        gt: Detections, hyp: Detections
    ) -> tuple[types.DictType, types.DictType]:
        assert gt.image_names is not None
        assert hyp.image_names is not None

        gt_img_dict = {name: i for i, name in enumerate(gt.image_names)}
        hyp_img_dict = {name: i for i, name in enumerate(hyp.image_names)}

        # Old gt ind -> new gt ind
        common_names = set(gt.image_names).intersection(hyp.image_names)
        gt_ind_dict = {gt_img_dict[name]: hyp_img_dict[name] for name in common_names}

        # For unmatched gts
        unmatched_gt_names = set(gt.image_names) - set(hyp.image_names)
        for name, i in zip(
            unmatched_gt_names,
            range(len(hyp_img_dict), len(hyp_img_dict) + len(unmatched_gt_names)),
        ):
            gt_ind_dict[gt_img_dict[name]] = i

        new_gt_image_ind_dict = nb.typed.Dict.empty(
            key_type=nb.int32, value_type=types.Tuple((types.int32, types.int32))
        )
        for old_ind, new_ind in gt_ind_dict.items():
            t1, t2 = gt.image_ind_dict[old_ind]
            new_gt_image_ind_dict[nb.int32(new_ind)] = (nb.int32(t1), nb.int32(t2))

        new_hyp_image_ind_dict = nb.typed.Dict.empty(
            key_type=nb.int32, value_type=types.Tuple((types.int32, types.int32))
        )
        for key, val in hyp.image_ind_dict.items():
            new_hyp_image_ind_dict[nb.int32(key)] = (nb.int32(val[0]), nb.int32(val[1]))

        return new_gt_image_ind_dict, new_hyp_image_ind_dict

    @staticmethod
    def _compute_ious(
        hyp_bboxes: npt.NDArray[np.float32],
        gt_bboxes: npt.NDArray[np.float32],
        hyp_image_ind_dict: dict[int, tuple[int, int]],
        gt_image_ind_dict: dict[int, tuple[int, int]],
    ) -> types.DictType:
        ious = nb.typed.Dict.empty(key_type=nb.int32, value_type=nb.float32[:, ::1])
        common_images = set(hyp_image_ind_dict.keys()).intersection(
            gt_image_ind_dict.keys()
        )

        for img_id in common_images:
            hyp_start, hyp_end = hyp_image_ind_dict[img_id]
            hyp_img_bboxes = hyp_bboxes[hyp_start:hyp_end]

            gt_start, gt_end = gt_image_ind_dict[img_id]
            gt_img_bboxes = gt_bboxes[gt_start:gt_end]

            ious[nb.int32(img_id)] = 1 - iou_dist(hyp_img_bboxes, gt_img_bboxes)

        return ious
