from typing import Literal, Mapping, Optional, TypedDict
import numba
import numpy as np
import numpy.typing as npt
from numba import types

from evaldet import Detections
from evaldet.utils.confusion_matrix import confusion_matrix as cm
from evaldet.utils.prec_recall import prec_recall_curve

from .base import DetMetricBase


class COCOResult(TypedDict):
    ap: Optional[float]
    recall: Optional[float]
    precision: Optional[float]
    n_gts: int


class COCOResults(COCOResult):
    class_results: dict[str, COCOResult]


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
    types.Tuple((types.int32[::1], types.bool_[::1], types.bool_[::1]))(
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
def evaluate_dataset(
    preds_bbox: npt.NDArray[np.float32],
    gts_bbox: npt.NDArray[np.float32],
    ious: dict[int, npt.NDArray[np.float32]],
    preds_conf: npt.NDArray[np.float32],
    img_ind_dict_preds: dict[int, tuple[int, int]],
    img_ind_dict_gts: dict[int, tuple[int, int]],
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

    keys_preds = set(img_ind_dict_preds.keys())
    keys_gts = set(img_ind_dict_gts.keys())
    all_imgs = list(keys_preds.union(keys_gts))

    for i in numba.prange(len(all_imgs)):
        img = all_imgs[i]

        default_inds = (numba.int32(0), numba.int32(0))
        preds_start, preds_end = img_ind_dict_preds.get(img, default_inds)
        gts_start, gts_end = img_ind_dict_gts.get(img, default_inds)

        img_ious = np.zeros((0, 0), dtype=np.float32)
        if img in ious:
            img_ious = ious[img]

        (matched_img, ignored_dets_img, ignored_gts_img, _) = evaluate_image(
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


class COCOMetrics(DetMetricBase):
    """
    Class for computing COCO metrics.

    The metrics are based on the [detection metrics](https://cocodataset.org/#detection-eval)
    of the COCO project. The metrics implemented here replicate the matching mechanism
    of the official ones, with the main difference being that the `"crowd"` attribute
    of ground truth objects is not taken into account here.

    Apart from computing AP and related metrics, you can also compute the confusion
    matrix, which is computed using the COCO matching mechanism, but ignoring the class
    information (assuming all objects are of the same class).
    """

    def __init__(self, ap_interpolation: Literal["coco", "pascal"] = "coco") -> None:
        """
        Initialize the object

        Args:
            ap_interpolation: The method to use for interpolating the average precision
                curve. The `coco` method uses 101 equally spaced point, while the
                `pascal` method directly computes the area under the (max) curve. In
                practice the difference between the two methods will be minimal.
        """
        if ap_interpolation not in ("coco", "pascal"):
            raise ValueError(
                "Only 'coco' and 'pascal' ap_interpolation values are allowed."
            )

        self.ap_interpolation = ap_interpolation

    @staticmethod
    def _check_compatibility(gt: Detections, hyp: Detections) -> None:
        if gt.image_names is None:
            raise ValueError("`image_names` must be provided for ground truths")

        if hyp.image_names is None:
            raise ValueError("`image_names` must be provided for ground truths")

        if gt.class_names is None:
            raise ValueError("`class_names` must be provided for ground truths")

        if hyp.class_names is None:
            raise ValueError("`class_names` must be provided for hypotheses")

        if hyp.class_names != gt.class_names:
            raise ValueError(
                "`class_names` must be the same for ground truths and hypotheses"
            )

        if hyp.confs is None:
            raise ValueError("`confs` must be provided for hypotheses")

        # TODO: what else

    def _compute_ap(
        self, precision: npt.NDArray[np.float64], recall: npt.NDArray[np.float64]
    ) -> float:
        if self.ap_interpolation == "coco":
            pass
        elif self.ap_interpolation == "pascal":
            pass

    # def

    def compute_metrics(
        self,
        gt: Detections,
        hyp: Detections,
        iou_threshold: float,
        area_range: tuple[float, float] = (0.0, float("inf")),
    ) -> COCOResults:
        self._check_compatibility(gt, hyp)

        class_results: dict[str, COCOResult] = {}

        assert gt.class_names is not None  # keep mypy happy
        for i, cls_name in enumerate(gt.class_names):
            hyp_cls = hyp.filter(hyp.classes == i)
            gt_cls = gt.filter(gt.classes == i)

            gt_img_ind_dict, hyp_img_ind_dict = self._match_images(gt_cls, hyp_cls)
            ious = self._compute_ious(
                hyp_cls.bboxes, gt_cls.bboxes, hyp_img_ind_dict, gt_img_ind_dict
            )

            hyp_matched, hyp_ignored, gts_ignored = evaluate_dataset(
                preds_bbox=hyp_cls.bboxes,
                gts_bbox=gt_cls.bboxes,
                ious=ious,
                preds_conf=hyp_cls.confs,
                img_ind_dict_preds=hyp_img_ind_dict,
                img_ind_dict_gts=gt_img_ind_dict,
                area_range=area_range,
                iou_threshold=iou_threshold,
            )
            n_gts = len(gt_cls.bboxes) - gts_ignored.sum()

            assert hyp_cls.confs is not None  # keep mypy happy
            pr_curve = prec_recall_curve(
                hyp_matched[~hyp_ignored], hyp_cls.confs[~hyp_ignored], n_gts
            )

            if pr_curve is None:
                class_results[cls_name] = dict(
                    ap=None, recall=None, precision=None, n_gts=n_gts
                )
            else:
                precision, recall = pr_curve
                ap = self._compute_ap(precision, recall)

                class_results[cls_name] = dict(
                    ap=ap, recall=recall[-1], precision=precision[-1], n_gts=n_gts
                )

        # Aggregate metrics
        total_gts = sum(res["n_gts"] for res in class_results.values())
        n_present_classes = sum(
            1 for res in class_results.values() if res["ap"] is not None
        )

        def sum_vals(d: dict, key: str) -> float:
            return sum(res[key] for res in d.values() if res[key] is not None)

        avg_ap = sum_vals(class_results, "ap") / n_present_classes
        w_avg_prec = sum_vals(class_results, "precision") / total_gts
        w_avg_rec = sum_vals(class_results, "recall") / total_gts

        metrics: COCOResults = dict(
            ap=avg_ap,
            precision=w_avg_prec,
            recall=w_avg_rec,
            n_gts=total_gts,
            class_results=class_results,
        )
        return metrics

    def compute_coco_summary(
        self,
        gt: Detections,
        hyp: Detections,
        iou_thresholds: tuple[float, ...] = (0.5, 0.75),
        sizes: Mapping[str, tuple[float, float]] = {"A": (0.0, 1.0)},
    ):
        pass

    def confusion_matrix(
        self,
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
            area_threshold: A tuple of `(lower_limit, upper_limit)` floats that set
                the lower and upper threshold for bound box area - detections with area
                outside of this range will be ignored.

        Returns:
            A `[C + 1, C + 1]` matrix, where `C` is the number of classes, and the entry
            at `[i, j]` denotes the number of hypotheses with class index `i` that were
            matched to a ground truth with the class index `j`. If the row or column
            index is `C` (last one), then this corresponds to the number of hypotheses
            or ground truths, respectively, that were not matched.
        """
        self._check_compatibility(gt, hyp)

        gt_img_ind_dict, hyp_img_ind_dict = self._match_images(gt, hyp)
        ious = self._compute_ious(
            hyp.bboxes, gt.bboxes, hyp_img_ind_dict, gt_img_ind_dict
        )

        det_matched, det_ignored, gts_ignored = evaluate_dataset(
            preds_bbox=hyp.bboxes,
            gts_bbox=gt.bboxes,
            ious=ious,
            preds_conf=hyp.confs,
            img_ind_dict_preds=hyp_img_ind_dict,
            img_ind_dict_gts=gt_img_ind_dict,
            area_range=area_range,
            iou_threshold=iou_threshold,
        )

        det_matched = det_matched[~det_ignored]
        det_classes = hyp.classes[~det_ignored]

        assert gt.class_names is not None  # keep mypy happy
        n_classes = len(gt.class_names)
        confusion_matrix = cm(
            det_matched, gts_ignored, det_classes, gt.classes, n_classes
        )

        return confusion_matrix
