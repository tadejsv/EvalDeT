import numba
import numpy as np
import numpy.typing as npt
from numba import types


@numba.njit(
    "Tuple((int32[:], bool_[:], bool_[:], int32))(float32[:,::1], float32[:,::1], float32[:,::1], float32[:], Tuple((float32, float32)), float32)",
)
def evaluate_image(
    preds_bbox: npt.NDArray[np.float32],
    gts_bbox: npt.NDArray[np.float32],
    ious: npt.NDArray[np.float32],
    preds_conf: npt.NDArray[np.float32],
    area_range: tuple[float, float],
    iou_threshold: float,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_], npt.NDArray[np.bool_], int]:
    """
    Evaluate a single image by matching predictions and ground truths for it.

    The matching process is the following:
    * First, ground truths are split into normal and ignored ones (those with area
      outside `area_range`)
    * In descending order according to prediction confidence, each prediction is matched
      with a ground truth that has not been matched to a previous prediction, and that
      has the highest IoU with it - the IoU needs to be above `iou_threshold`. If no
      match can be made with a normal ground truth, match is attempted with an ignored
      ground truth.
    * Finally, the predictions that were matched to an ignored ground truth, or those
      that were unmatched and had area outside of `area_range`, are marked as ignored.

    Args:
        preds_bbox: A `[N, 4]` array of prediction bounding boxes in xywh format
        gts_bbox: A `[M, 4]` array of ground truth bounding boxes in xywh format
        ious: A `[N, M]` array of IoU similarities between predictions and ground truths
        preds_conf: A `[N, ]` array of confidence scores of predictions
        area_range: A tuple `(lower_limit, upper_limit)` of limits for bounding box
            areas. Ground truths with area outside of this range will be ignored, as
            will the predictions matched to them and unmatched predictions with area
            outside of this range
        iou_threshold: The lower threshold for an IOU between a prediction and a
            bounding box that is required for establishing a match

    Returns:
        A tuple with 4 elements:
            - an integer array of shape `[N, ]`, indicating the index of the ground
              truth that the prediction was matched to. Unmatched predictions will have
              the matched index of `-1`.
            - a boolean array of shape `[N, ]`, indicating whether the prediction was
              ignored or not
            - a boolean array of shape `[M, ]`, indicating whether the ground truth was
              ignored or not
            - an integer denoting the number of non-ignored ground truths in the image
    """
    matched = np.full(preds_conf.shape, -1, dtype=np.int32)

    # Compute area
    gts_area = gts_bbox[:, 2] * gts_bbox[:, 3]
    preds_area = preds_bbox[:, 2] * preds_bbox[:, 3]

    ignore_preds_area = np.logical_or(
        preds_area < area_range[0], preds_area > area_range[1]
    )

    # Sort gts by ignore
    gts_ignore_orig = np.logical_or(gts_area < area_range[0], gts_area > area_range[1])
    sort_gt = np.argsort(gts_ignore_orig)
    gts_ignore = gts_ignore_orig[sort_gt]
    gts_area = gts_area[sort_gt]
    gts_bbox = gts_bbox[sort_gt]
    n_gts = int((~gts_ignore).sum())

    if preds_conf.size == 0 or gts_bbox.size == 0:
        return (matched, ignore_preds_area, gts_ignore_orig, n_gts)

    # Sort preds by conf
    sort_preds = np.argsort(-preds_conf)
    preds_bbox = preds_bbox[sort_preds]
    preds_conf = preds_conf[sort_preds]
    ignore_preds_area = ignore_preds_area[sort_preds]

    # Get the index where to split gts into normal and ignore
    start_ignore = int(n_gts)
    if start_ignore == 0 and gts_ignore[0] == 0:
        start_ignore = len(gts_area)  # Nothing is ignored

    ious = ious[sort_preds, :][:, sort_gt]

    ignore_preds = np.zeros(preds_conf.shape, dtype=np.bool_)

    for p_ind in range(matched.shape[0]):
        # Try mathing with normal gts
        if start_ignore > 0:
            best_match_ind = np.argmax(ious[p_ind, :start_ignore])
            if ious[p_ind, best_match_ind] > iou_threshold:
                ious[:, best_match_ind] = -1
                matched[p_ind] = sort_gt[best_match_ind]
                continue

        # Try matching with ignored gts
        if start_ignore < len(gts_area):
            ignore_match_ind = np.argmax(ious[p_ind, start_ignore:]) + start_ignore
            if ious[p_ind, ignore_match_ind] > iou_threshold:
                ious[:, ignore_match_ind] = -1
                matched[p_ind] = sort_gt[ignore_match_ind]
                ignore_preds[p_ind] = True

    ignore_preds = np.logical_or(
        ignore_preds, np.logical_and(matched == -1, ignore_preds_area)
    )

    sort_back_preds = np.argsort(sort_preds)
    matched_orig = matched[sort_back_preds]
    ignore_preds_orig = ignore_preds[sort_back_preds]

    return matched_orig, ignore_preds_orig, gts_ignore_orig, n_gts


@numba.njit(
    types.float32[:, ::1](
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.DictType(types.int32, types.float32[:, ::1]),
        types.float32[::1],
        types.DictType(types.int32, types.Tuple((types.int32, types.int32))),
        types.DictType(types.int32, types.Tuple((types.int32, types.int32))),
        types.Tuple((types.float32, types.float32)),
        types.float32,
    ),
    parallel=True,
)
def calculate_pr_curve(
    preds_bbox: npt.NDArray[np.float32],
    gts_bbox: npt.NDArray[np.float32],
    ious: dict[int, npt.NDArray[np.float32]],
    preds_conf: npt.NDArray[np.float32],
    img_ind_dict_preds: dict[int, tuple[int, int]],
    img_ind_dict_gts: dict[int, tuple[int, int]],
    area_range: tuple[float, float],
    iou_threshold: float,
) -> npt.NDArray[np.float32]:
    """
    Calculate the precision-recall curve.

    Args:
        preds_bbox: A `[N, 4]` array of prediction bounding boxes in xywh format
        gts_bbox: A `[M, 4]` array of ground truth bounding boxes in xywh format
        ious: A dict with keys being image indices, and values beind 2D numpy arrays,
          containing IoU similarity scores between predictions and ground truths for
          each image.
        preds_conf: A `[N,]` array of prediction confidence scores.
        img_ind_dict_preds: A dictionary where keys are image indices, and values are
          a tuple of 2 integers, denoting the starting and ending position of the
          image's detection bounding boxes and confidences in `preds_bbox` and
          `preds_conf`, respectively.
        img_ ind_dict_gts: A dictionary where keys are image indices, and values are
          a tuple of 2 integers, denoting the starting and ending position of the
          image's ground truth bounding boxes `preds_bbox`.
        area_range: A tuple `(lower_limit, upper_limit)` of limits for bounding box
            areas. Ground truths with area outside of this range will be ignored, as
            will the predictions matched to them and unmatched predictions with area
            outside of this range
        iou_threshold: The lower threshold for an IOU between a prediction and a
            bounding box that is required for establishing a match

    Returns:
        An array of size `[2, K]`, where each column is a point on the precision-recall
        curve (first row consists of precision values, the second of recall)
    """
    det_ignored = np.zeros_like(preds_conf, dtype=np.bool_)
    det_matched = np.zeros_like(preds_conf, dtype=np.int32)
    n_gts = 0

    keys_preds = set(img_ind_dict_preds.keys())
    keys_gts = set(img_ind_dict_gts.keys())
    all_imgs = list(keys_preds.union(keys_gts))

    for i in numba.prange(len(all_imgs)):
        img = all_imgs[i]
        if img in img_ind_dict_preds:
            preds_start, preds_end = img_ind_dict_preds[img]
        else:
            preds_start, preds_end = 0, 0

        if img in img_ind_dict_gts:
            gts_start, gts_end = img_ind_dict_gts[img]
        else:
            gts_start, gts_end = 0, 0

        img_ious = np.zeros((0, 0), dtype=np.float32)
        if img in ious:
            img_ious = ious[img]

        (matched_img, ignored_dets_img, _, n_gts_img) = evaluate_image(
            preds_bbox[preds_start:preds_end],
            gts_bbox[gts_start:gts_end],
            img_ious,
            preds_conf=preds_conf[preds_start:preds_end],
            area_range=area_range,
            iou_threshold=iou_threshold,
        )
        n_gts += n_gts_img

        if preds_end > 0:
            det_ignored[preds_start:preds_end] = ignored_dets_img
            det_matched[preds_start:preds_end] = matched_img

    det_matched = det_matched[~det_ignored]
    preds_conf = preds_conf[~det_ignored]

    sort_arr = np.argsort(-preds_conf)
    preds_conf = preds_conf[sort_arr]
    det_matched_bool = det_matched[sort_arr] > -1

    precision = np.cumsum(det_matched_bool).astype(np.float32) / np.arange(
        1, len(det_matched_bool) + 1
    ).astype(np.float32)
    recall = np.cumsum(det_matched_bool).astype(np.float32) / numba.float32(n_gts)

    prec_recall = np.stack((precision, recall))

    return prec_recall
