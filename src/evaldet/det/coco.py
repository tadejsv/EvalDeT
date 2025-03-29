"""COCO object detection metrics."""

from collections.abc import Mapping
from enum import Enum
from typing import Any, TypedDict, cast

import numba
import numpy as np
import numpy.typing as npt
from numba import types

from evaldet import Detections
from evaldet.utils.prec_recall import prec_recall_curve

from . import utils


def _nonemean(x: npt.NDArray[np.float32]) -> float | None:
    if x.size == 0 or np.all(np.isnan(x)):
        return None

    return float(np.nanmean(x))


_BASE_50_95_THRESHOLDS = (
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
)


class COCOResult(TypedDict):
    """
    A typed dictionary for storing the results of the per-class COCO metric evaluation.
    """

    ap: float | None
    recall: float | None
    precision: float | None
    n_gts: int


class COCOResults(TypedDict):
    """
    A typed dictionary for storing the results of the COCO metric evaluation
    (across all classes).
    """

    mean_ap: float | None
    class_results: dict[str, COCOResult]


class COCOSummaryResults(TypedDict):
    """
    A typed dictionary for storing the COCO summary results.
    """

    mean_ap: float | None
    ap_50: float | None
    ap_75: float | None

    mean_ap_per_class: dict[str, float | None]
    ap_50_per_class: dict[str, float | None]
    ap_75_per_class: dict[str, float | None]

    mean_ap_sizes: dict[str, float | None]
    mean_ap_sizes_per_class: dict[str, dict[str, float | None]]


class APInterpolation(str, Enum):
    """
    Enum for specifying the method of Average Precision (AP) interpolation.
    """

    coco = "coco"
    pascal = "pascal"


@numba.njit(  # type: ignore[misc]
    types.Tuple((types.int32[::1], types.bool_[::1], types.bool_[::1], types.int32))(
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.float32[::1],
        types.Tuple((types.float32, types.float32)),
        types.float32,
    ),
)
def _evaluate_image(
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

    ignore_preds_area = (preds_area < area_range[0]) | (preds_area > area_range[1])

    # Sort gts by ignore
    gts_ignore = (gts_area < area_range[0]) | (gts_area > area_range[1])
    sort_gt = np.argsort(gts_ignore)
    n_gts = int((~gts_ignore).sum())

    if preds_conf.size == 0 or gts_bbox.size == 0:
        return (matched, ignore_preds_area, gts_ignore, n_gts)

    # Sort preds by conf
    sort_preds = np.argsort(-preds_conf)

    ious = ious[:, sort_gt]

    ignore_preds = np.zeros_like(ignore_preds_area)

    for i in range(matched.shape[0]):
        p_ind = sort_preds[i]
        # Try mathing with normal gts
        if n_gts > 0:
            best_match_ind = np.argmax(ious[p_ind, :n_gts])
            if ious[p_ind, best_match_ind] > iou_threshold:
                ious[:, best_match_ind] = -1
                matched[p_ind] = sort_gt[best_match_ind]
                continue

        # Try matching with ignored gts
        if n_gts < len(gts_area):
            ignore_match_ind = np.argmax(ious[p_ind, n_gts:]) + n_gts
            if ious[p_ind, ignore_match_ind] > iou_threshold:
                ious[:, ignore_match_ind] = -1
                matched[p_ind] = sort_gt[ignore_match_ind]
                ignore_preds[p_ind] = True

    ignore_preds = ignore_preds | ((matched == -1) & (ignore_preds_area))

    return matched, ignore_preds, gts_ignore, n_gts


@numba.njit(  # type: ignore[misc]
    types.Tuple((types.int32[::1], types.bool_[::1], types.bool_[::1]))(
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.DictType(types.int64, types.float32[:, ::1]),
        types.float32[::1],
        types.int32[:, ::1],
        types.Tuple((types.float32, types.float32)),
        types.float32,
    ),
    parallel=False,
)
def _evaluate_dataset(
    preds_bbox: npt.NDArray[np.float32],
    gts_bbox: npt.NDArray[np.float32],
    ious: dict[int, npt.NDArray[np.float32]],
    preds_conf: npt.NDArray[np.float32],
    img_ind_corr: npt.NDArray[np.float32],
    area_range: tuple[float, float],
    iou_threshold: float,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Calculate the precision-recall curve.

    Args:
        preds_bbox: A `[N, 4]` array of prediction bounding boxes in xywh format
        gts_bbox: A `[M, 4]` array of ground truth bounding boxes in xywh format
        ious: A dict with keys being image indices, and values beind 2D numpy arrays,
          containing IoU similarity scores between predictions and ground truths for
          each image.
        preds_conf: A `[N,]` array of prediction confidence scores.
        img_ind_corr: An `[K, 4]` array, that for each image (in the union of images
            from both hypotheses and ground truths) contains the start and end indices
            for that image in both hypotheses and ground truths. A row is of the form
            `hyp_start, hyp_end, gt_start, gt_end`. If `hyp_end` or `gt_end` is 0, that
            means that this image does not appear in hypotheses or ground truths,
            respectivelly.
        area_range: A tuple `(lower_limit, upper_limit)` of limits for bounding box
            areas. Ground truths with area outside of this range will be ignored, as
            will the predictions matched to them and unmatched predictions with area
            outside of this range
        iou_threshold: The lower threshold for an IOU between a prediction and a
            bounding box that is required for establishing a match

    Returns:
        A tuple with 3 elements:
            - an integer array of shape `[N, ]`, indicating the index of the ground
              truth that the prediction was matched to. Unmatched predictions will have
              the matched index of `-1`.
            - a boolean array of shape `[N, ]`, indicating whether the prediction was
              ignored or not
            - a boolean array of shape `[M, ]`, indicating whether the ground truth was
              ignored or not

    """
    det_ignored = np.zeros_like(preds_conf, dtype=np.bool_)
    gts_ignored = np.zeros((gts_bbox.shape[0],), dtype=np.bool_)
    det_matched = np.full_like(preds_conf, -1, dtype=np.int32)

    for img in numba.prange(len(img_ind_corr)):
        preds_start, preds_end = img_ind_corr[img, :2]
        gts_start, gts_end = img_ind_corr[img, 2:]

        if preds_end == 0 and gts_end == 0:
            continue

        img_ious = np.zeros((0, 0), dtype=np.float32)
        if img in ious:
            img_ious = ious[img]

        (matched_img, ignored_dets_img, ignored_gts_img, _) = _evaluate_image(
            preds_bbox[preds_start:preds_end],
            gts_bbox[gts_start:gts_end],
            img_ious,
            preds_conf=preds_conf[preds_start:preds_end],
            area_range=area_range,
            iou_threshold=iou_threshold,
        )

        if preds_end > 0:
            det_ignored[preds_start:preds_end] = ignored_dets_img

            # Add gts_start as image offset
            matched_img = matched_img + numba.int32(gts_start)
            matched_img[matched_img < gts_start] = -1
            det_matched[preds_start:preds_end] = matched_img

        if gts_end > 0:
            gts_ignored[gts_start:gts_end] = ignored_gts_img

    return det_matched, det_ignored, gts_ignored


def _check_compatibility(gt: Detections, hyp: Detections) -> None:
    if hyp.class_names != gt.class_names:
        raise ValueError(
            "`class_names` must be the same for ground truths and hypotheses"
        )

    if hyp.confs is None:
        raise ValueError("`confs` must be provided for hypotheses")


def _compute_ap(
    precision: npt.NDArray[np.float64] | None,
    recall: npt.NDArray[np.float64] | None,
    ap_interpolation: APInterpolation = APInterpolation.coco,
) -> float | None:
    if precision is None or recall is None:
        return None

    precision = np.concatenate([[0], precision])
    recall = np.concatenate([[0], recall])
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    if ap_interpolation == "coco":
        thresholds = np.linspace(0.0, 1.00, 101, endpoint=True)
        r_inds = np.searchsorted(recall, thresholds, side="left")
        prec_a = np.concatenate([precision, [0]])
        return prec_a[r_inds].mean()  # type: ignore[no-any-return]

    if ap_interpolation == "pascal":
        return np.trapezoid(precision, recall)  # type: ignore[no-any-return]

    # No matching ap_interpolation
    raise ValueError(f"Unknown interpolation {ap_interpolation}")


def compute_metrics(
    gt: Detections,
    hyp: Detections,
    iou_threshold: float,
    area_range: tuple[float, float] = (0.0, float("inf")),
    ap_interpolation: APInterpolation = APInterpolation.coco,
) -> COCOResults:
    """
    Compute COCO metrics (mAP, precision and recall).

    The metrics are based on the [detection metrics](https://cocodataset.org/#detection-eval)
    of the COCO project. The metrics implemented here replicate the matching mechanism
    of the official ones, with the main difference being that the `"crowd"` attribute
    of ground truth objects is not taken into account here.

    Args:
        gt: Ground truth detections.
        hyp: Hypotheses detections.
        iou_threshold: IoU threshold for matching.
        area_range: The upper and lower threshold for object area. Objects outside
            of this range will be ignored.
        ap_interpolation: The method to use for interpolating the average precision
            curve. The `coco` method uses 101 equally spaced point, while the
            `pascal` method directly computes the area under the (max) curve. In
            practice the difference between the two methods will be minimal.

    Returns:
        A dictionary that contains the mean average precision metrics for the whole
        dataset (averaged class APs), as well as the results for each class (AP,
        precision and recall).

    """
    _check_compatibility(gt, hyp)

    class_results: dict[str, COCOResult] = {}

    for i, cls_name in enumerate(gt.class_names):
        hyp_cls = hyp.filter(hyp.classes == i)
        gt_cls = gt.filter(gt.classes == i)

        img_ind_corr = utils.match_images(gt_cls, hyp_cls)
        ious = utils.compute_ious(hyp_cls.bboxes, gt_cls.bboxes, img_ind_corr)

        hyp_matched, hyp_ignored, gts_ignored = _evaluate_dataset(
            preds_bbox=hyp_cls.bboxes,
            gts_bbox=gt_cls.bboxes,
            ious=ious,
            preds_conf=hyp_cls.confs,
            img_ind_corr=img_ind_corr,
            area_range=area_range,
            iou_threshold=iou_threshold,
        )
        n_gts = len(gt_cls.bboxes) - gts_ignored.sum()

        if hyp_cls.confs is None:
            msg = "hyp_cls.confs should not be None"
            raise ValueError(msg)

        precision, recall = prec_recall_curve(
            hyp_matched[~hyp_ignored] != -1, hyp_cls.confs[~hyp_ignored], n_gts
        )

        if precision is None or recall is None:
            class_results[cls_name] = {
                "ap": None,
                "recall": None,
                "precision": None,
                "n_gts": n_gts,
            }
        else:
            ap = _compute_ap(precision, recall, ap_interpolation)

            class_results[cls_name] = {
                "ap": ap,
                "recall": recall[-1],
                "precision": precision[-1],
                "n_gts": n_gts,
            }

    # Aggregate metrics
    aps = np.array([res["ap"] for res in class_results.values()], dtype=float)
    mean_ap = _nonemean(aps)
    metrics: COCOResults = {"mean_ap": mean_ap, "class_results": class_results}
    return metrics


_IOU_50_INDEX = 0
_IOU_75_INDEX = 5


def compute_coco_summary(
    gt: Detections,
    hyp: Detections,
    sizes: Mapping[str, tuple[float, float]] = {
        "small": (0.0, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, float("inf")),
    },
    ap_interpolation: APInterpolation = APInterpolation.coco,
) -> COCOSummaryResults:
    """
    Compute COCO summary metrics.

    This computes the standard COCO summary metrics:
    * mAP (overall, and for each class)
    * mAP at IoU threshold of 0.5 and 0.75 (overall and for each class)
    * mAP for different sizes of objects (overall and for each class)

    The metrics are based on the [detection metrics](https://cocodataset.org/#detection-eval)
    of the COCO project. The metrics implemented here replicate the matching mechanism
    of the official ones, with the main difference being that the `"crowd"` attribute
    of ground truth objects is not taken into account here.

    Computing the summary metrics via this method is more efficient than using
    `compute_metrics` to compute them individually, as some intermediate steps
    (matching images, computing IoUs) can be shared between metrics.

    Args:
        gt: Ground truth detections
        hyp: Hypotheses detections
        sizes: A dictionary with size names as keys and their area ranges as values.
        ap_interpolation: The method to use for interpolating the average precision
            curve. The `coco` method uses 101 equally spaced point, while the
            `pascal` method directly computes the area under the (max) curve. In
            practice the difference between the two methods will be minimal.

    Returns:
        A dictionary with all overall and per class summary metrics.

    """
    _check_compatibility(gt, hyp)

    sizes_all = {"_all": (0.0, float("inf"))} | sizes  # type: ignore[operator]
    ap_results: dict[tuple[str, str, int], float | None] = {}

    for i_cls, cls_name in enumerate(gt.class_names):
        hyp_cls = hyp.filter(hyp.classes == i_cls)
        gt_cls = gt.filter(gt.classes == i_cls)

        img_ind_corr = utils.match_images(gt_cls, hyp_cls)

        ious = utils.compute_ious(hyp_cls.bboxes, gt_cls.bboxes, img_ind_corr)

        for i_thr, iou_threshold in enumerate(_BASE_50_95_THRESHOLDS):
            for size_name, area_range in sizes_all.items():
                hyp_matched, hyp_ignored, gts_ignored = _evaluate_dataset(
                    preds_bbox=hyp_cls.bboxes,
                    gts_bbox=gt_cls.bboxes,
                    ious=ious,
                    preds_conf=hyp_cls.confs,
                    img_ind_corr=img_ind_corr,
                    area_range=area_range,
                    iou_threshold=iou_threshold,
                )
                n_gts = len(gt_cls.bboxes) - gts_ignored.sum()

                if hyp_cls.confs is None:
                    msg = "hyp_cls.confs should not be None"
                    raise ValueError(msg)

                precision, recall = prec_recall_curve(
                    hyp_matched[~hyp_ignored] != -1,
                    hyp_cls.confs[~hyp_ignored],
                    n_gts,
                )

                ap_results[(cls_name, size_name, i_thr)] = _compute_ap(
                    precision, recall, ap_interpolation
                )

    # Aggregate metricss
    aps = np.array(list(ap_results.values()), dtype=np.float64)
    cls_arr = np.array([k[0] for k in ap_results])
    size_arr = np.array([k[1] for k in ap_results])
    iou_t_arr = np.array([k[2] for k in ap_results])

    results: dict[str, Any] = {}
    results["mean_ap"] = _nonemean(aps[size_arr == "_all"])
    results["ap_50"] = _nonemean(
        aps[(iou_t_arr == _IOU_50_INDEX) & (size_arr == "_all")]
    )
    results["ap_75"] = _nonemean(
        aps[(iou_t_arr == _IOU_75_INDEX) & (size_arr == "_all")]
    )

    results["mean_ap_per_class"] = {}
    results["ap_50_per_class"] = {}
    results["ap_75_per_class"] = {}

    for cls in gt.class_names:
        results["mean_ap_per_class"][cls] = _nonemean(
            aps[(size_arr == "_all") & (cls_arr == cls)]
        )
        results["ap_50_per_class"][cls] = _nonemean(
            aps[(size_arr == "_all") & (cls_arr == cls) & (iou_t_arr == _IOU_50_INDEX)]
        )
        results["ap_75_per_class"][cls] = _nonemean(
            aps[(size_arr == "_all") & (cls_arr == cls) & (iou_t_arr == _IOU_75_INDEX)]
        )

    results["mean_ap_sizes"] = {}
    for size_name in sizes:
        results["mean_ap_sizes"][size_name] = _nonemean(aps[size_arr == size_name])

    results["mean_ap_sizes_per_class"] = {}
    for cls in gt.class_names:
        results["mean_ap_sizes_per_class"][cls] = {}
        for size_name in sizes:
            results["mean_ap_sizes_per_class"][cls][size_name] = _nonemean(
                aps[(size_arr == size_name) & (cls_arr == cls)]
            )

    return cast("COCOSummaryResults", results)


def confusion_matrix(
    gt: Detections,
    hyp: Detections,
    iou_threshold: float,
    area_range: tuple[float, float] = (0.0, float("inf")),
) -> npt.NDArray[np.int32]:
    """
    Compute the confusion matrix.

    This method computes the confusion matrix, which matches objects in hypotheses
    and ground truth in a class-agnostic way, and then computes the number of
    objects in hypotheses of class X that were matched with object of class Y in
    ground truths. Number of unmatched objects in hypotheses and ground truths is
    also calculated.

    Args:
        gt: Ground truth detections. It must contain `image_names` property, which
            is used to match the images between `gt` and `hyp`. If it contains
            `class_names`, they must match the ones from `hyp`.
        hyp: Hypotheses (prediction) detections. It must contain `image_names`
            property, which is used to match the images between `gt` and `hyp`. If
            it contains `class_names`, they must match the ones from `gt`.
        iou_threshold: IoU threshold for matching hypotheses objects to ground
            truth.
        area_range: A tuple of `(lower_limit, upper_limit)` floats that set
            the lower and upper threshold for bound box area - detections with area
            outside of this range will be ignored.

    Returns:
        A `[C + 1, C + 1]` matrix, where `C` is the number of classes, and the entry
        at `[i, j]` denotes the number of hypotheses with class index `i` that were
        matched to a ground truth with the class index `j`. If the row or column
        index is `C` (last one), then this corresponds to the number of hypotheses
        or ground truths, respectively, that were not matched.

    """
    _check_compatibility(gt, hyp)

    img_ind_corr = utils.match_images(gt, hyp)
    ious = utils.compute_ious(hyp.bboxes, gt.bboxes, img_ind_corr)

    det_matched, det_ignored, gts_ignored = _evaluate_dataset(
        preds_bbox=hyp.bboxes,
        gts_bbox=gt.bboxes,
        ious=ious,
        preds_conf=hyp.confs,
        img_ind_corr=img_ind_corr,
        area_range=area_range,
        iou_threshold=iou_threshold,
    )

    det_matched = det_matched[~det_ignored]
    det_classes = hyp.classes[~det_ignored]

    n_classes = len(gt.class_names)
    return utils.confusion_matrix(  # type: ignore[no-any-return]
        det_matched, gts_ignored, det_classes, gt.classes, n_classes
    )
