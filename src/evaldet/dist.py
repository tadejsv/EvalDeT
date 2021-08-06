import numpy as np


def _box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_intersection(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    max_xmin = np.maximum(boxes_1[:, 0], boxes_2[:, 0])
    min_xmax = np.minimum(boxes_1[:, 2], boxes_2[:, 2])
    max_ymin = np.maximum(boxes_1[:, 1], boxes_2[:, 1])
    min_ymax = np.minimum(boxes_1[:, 3], boxes_2[:, 3])
    x_intersection = (min_xmax - max_xmin).clip(0)
    y_intersection = (min_ymax - max_ymin).clip(0)

    return x_intersection * y_intersection


def iou_dist(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Computes a matrix of IoU distances.

    This computes a matrix of IoU distances, representing IoU distances
    for all combinations of boxes from the first set with a box from the
    second set.

    The returned matrix contains IoU distances - which equal 1 minus the IoU scores.

    Args:
        boxes_1: The first set of boxes, should be an array of shape ``[N, 4]``,
            where the boxes are in format ``[xmin, ymin, xmax, ymax]``
        boxes_2: The first set of boxes, should be an array of shape ``[M, 4]``,
            where the boxes are in format ``[xmin, ymin, xmax, ymax]``
    Returns:
        An array of shape ``[N, M]``, where the entry at index ``[i, j]`` represents
        the IoU distance between the i-th box from ``boxes_1`` and j-th box from
        ``boxes_2``
    """
    rep_boxes_1 = np.tile(boxes_1, (boxes_2.shape[0], 1))
    rep_boxes_2 = boxes_2.repeat(boxes_1.shape[0], axis=0)
    iou_long = iou_dist_pairwise(rep_boxes_1, rep_boxes_2)
    iou_matrix = iou_long.reshape(boxes_1.shape[0], boxes_2.shape[0], order="F")

    return iou_matrix


def iou_dist_pairwise(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU distance between two sets of boxes.

    The returned array contains IoU distances - which equal 1 minus the IoU scores.

    Args:
        boxes_1: The first set of boxes, should be an array of shape ``[N, 4]``,
            where the boxes are in format ``[xmin, ymin, xmax, ymax]``
        boxes_2: The first set of boxes, should be an array of shape ``[N, 4]``,
            where the boxes are in format ``[xmin, ymin, xmax, ymax]``
    Returns:
        An array of shape ``[N, ]``, where the entry at index ``[i]`` represents
        the IoU distance between the i-th box from ``boxes_1`` and i-th box from
        ``boxes_2``
    """
    intersection = _box_intersection(boxes_1, boxes_2)
    union = _box_area(boxes_1) + _box_area(boxes_2) - intersection
    iou = 1 - intersection / union

    return iou
