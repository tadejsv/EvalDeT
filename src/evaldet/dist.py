import numpy as np


def iou_dist(bboxes_1: np.ndarray, bboxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoU distance between two batches of bounding boxes.

    The IoU distance is computed as 1 minus the IoU similarity.

    Args:
        bboxes_1: An ``[N, 4]`` array of bounding boxes in the xywh format.
        bboxes_2: An ``[M, 4]`` array of bounding boxes in the xywh format.

    Returns:
        An ``[N, M]`` array of pairwise IoU distances.
    """

    b1, b2 = np.expand_dims(bboxes_1, 1), np.expand_dims(bboxes_2, 0)

    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 0] + b1[..., 2], b2[..., 0] + b2[..., 2])
    yy2 = np.minimum(b1[..., 1] + b1[..., 3], b2[..., 1] + b2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    union = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - intersection
    iou = intersection / union

    return 1 - iou
