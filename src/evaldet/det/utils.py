"""Various utilitis for computing object detection metrics."""

import numba as nb
import numpy as np
import numpy.typing as npt
from numba import types

from evaldet import Detections
from evaldet.dist import iou_dist


@nb.njit(  # type: ignore[misc]
    types.DictType(types.int64, types.float32[:, ::1])(
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.int32[:, ::1],
    )
)
def compute_ious(
    hyp_bboxes: npt.NDArray[np.float32],
    gt_bboxes: npt.NDArray[np.float32],
    image_ind_corr: npt.NDArray[np.int32],
) -> types.DictType:
    """
    Compute the IoUs for each image.

    This computes the IoUs between bounding boxes in hypotheses and ground truth for
    each image present in both

    Args:
        hyp_bboxes: An `[N,4]` array of hypotheses bounding boxes, where each row is of
            the format `xywh`.
        gt_bboxes: An `[M,4]` array of ground truth bounding boxes, where each row is of
            the format `xywh`.
        image_ind_corr: An `[K,4]` array where each row is of the form
            `[hyp_start, hyp_end, gt_start, gt_end]`. The start/end indices indicate at
            which index do elements of the frame start or end in the bounding box array,
            for hypotheses and indices. If the end index is 0, this means that this
            frame is not present in hypotheses/ground truths

    Returns:
        A numba dictionary where key is the image index, corresponding to the row index
        in `image_ind_corr`, and the key is a `[H, G]` array of IoUs, where rows
        correspond to hypotheses and coulmns to ground truths on that image.

    """
    ious = {}
    common_images = np.nonzero(image_ind_corr[:, 3] * image_ind_corr[:, 1])[0]

    for img_id in common_images:
        hyp_start, hyp_end = image_ind_corr[img_id, :2]
        hyp_img_bboxes = hyp_bboxes[hyp_start:hyp_end]

        gt_start, gt_end = image_ind_corr[img_id, 2:]
        gt_img_bboxes = gt_bboxes[gt_start:gt_end]

        ious[img_id] = 1 - iou_dist(hyp_img_bboxes, gt_img_bboxes)

    return ious


def match_images(gt: Detections, hyp: Detections) -> npt.NDArray[np.int32]:
    """
    Match images between ground truths and hypotheses.

    This creates an array with the start and end indices of the same image for
    detections and hypotheses.

    Args:
        gt: Ground truth detections
        hyp: Hypotheses (predictions) detections

    Returns:
        An array of shape `[N, 4]`, where each row is of the form
        `[hyp_start, hyp_end, gt_start, gt_end]`. The start/end indices indicate at
        which index do elements of the frame start. or end (for example for bboxes),
        for hypotheses and indices. If the end index is 0, this means that this frame is
        not present in hypotheses/ground truths

    """
    gt_img_dict = {name: i for i, name in enumerate(gt.image_names)}
    hyp_img_dict = {name: i for i, name in enumerate(hyp.image_names)}

    all_images = sorted(set(gt.image_names).union(hyp.image_names))
    image_ind_corr = np.zeros((len(all_images), 4), dtype=np.int32)

    for i, img in enumerate(all_images):
        if img in gt_img_dict:
            img_id = gt_img_dict[img]
            if img_id in gt.image_ind_dict:
                image_ind_corr[i, 2:] = gt.image_ind_dict[img_id]

        if img in hyp_img_dict:
            img_id = hyp_img_dict[img]
            if img_id in hyp.image_ind_dict:
                image_ind_corr[i, :2] = hyp.image_ind_dict[img_id]

    return image_ind_corr


@nb.njit(  # type: ignore[misc]
    nb.int32[:, ::1](
        nb.int32[::1],
        nb.boolean[::1],
        nb.int32[::1],
        nb.int32[::1],
        nb.int32,
    ),
)
def confusion_matrix(
    matching: npt.NDArray[np.int32],
    gt_ignored: npt.NDArray[np.bool_],
    hyp_classes: npt.NDArray[np.int32],
    gt_classes: npt.NDArray[np.int32],
    n_classes: int,
) -> npt.NDArray[np.int32]:
    """
    Compute the confusion matrix.

    This method computes the confusion matrix, showing the number of objects in
    hypotheses of class X that were matched with object of class Y in ground truths.

    Args:
        matching: A `[N,]` shaped integer array, where entry at `i`-th position denotes
            the index of the ground truth that the `i`-th hypothesis is matched with.
        gt_ignored: A `[M,]` array, denoting whether a ground truth is ignored or not.
        hyp_classes: A `[N,]` array, denoting the class index of each hypothesis.
        gt_classes: A `[M,]` array, denoting the class index of each ground truth.
        n_classes: Total number of classes.

    Returns:
        A `[n_classes + 1, n_classes + 1]` matrix, where the entry  at `[i, j]` denotes
        the number of hypotheses with class index `i` that were matched to a ground
        truth with the class index `j`. If the row or column index is `n_classes`
        (last one), then this corresponds to the number of hypotheses or ground truths,
        respectively, that were not matched.

    """
    conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int32)

    # First fill in totals for gts
    gt_non_ignored_classes = gt_classes[~gt_ignored]
    for i in range(gt_non_ignored_classes.shape[0]):
        conf_matrix[-1, gt_non_ignored_classes[i]] += 1

    for i in range(matching.shape[0]):
        r = hyp_classes[i]
        gt_ind = matching[i]

        if gt_ind > -1 and gt_ignored[gt_ind] is True:
            c = -1
        elif gt_ind > -1:
            c = gt_classes[gt_ind]

            # Subtract match from gt totals
            conf_matrix[-1, c] -= 1
        else:
            c = -1

        conf_matrix[r, c] += 1

    return conf_matrix
