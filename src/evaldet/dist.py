"""Module with functions for computing distances between two sets of bounding boxes."""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(  # type: ignore[misc]
    numba.float32[:, ::1](numba.float32[:, ::1], numba.float32[:, ::1]),
    fastmath=True,
    parallel=True,
)
def iou_dist(
    bboxes_1: npt.NDArray[np.float32], bboxes_2: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute the IoU distance between two batches of bounding boxes.

    The IoU distance is computed as 1 minus the IoU similarity.

    Args:
        bboxes_1: An `[N, 4]` array of bounding boxes in the xywh format.
        bboxes_2: An `[M, 4]` array of bounding boxes in the xywh format.

    Returns:
        An `[N, M]` array of pairwise IoU distances.

    """
    ious = np.zeros((bboxes_1.shape[0], bboxes_2.shape[0]), dtype=np.float32)

    for i in numba.prange(bboxes_1.shape[0]):
        for j in numba.prange(bboxes_2.shape[0]):
            xx1 = max(bboxes_1[i, 0], bboxes_2[j, 0])
            xx2 = min(bboxes_1[i, 0] + bboxes_1[i, 2], bboxes_2[j, 0] + bboxes_2[j, 2])
            yy1 = max(bboxes_1[i, 1], bboxes_2[j, 1])
            yy2 = min(bboxes_1[i, 1] + bboxes_1[i, 3], bboxes_2[j, 1] + bboxes_2[j, 3])

            zero = numba.float32(0)
            intersection = max(zero, xx2 - xx1) * max(zero, yy2 - yy1)
            if intersection > 0:
                union = (
                    bboxes_1[i, 2] * bboxes_1[i, 3]
                    + bboxes_2[j, 2] * bboxes_2[j, 3]
                    - intersection
                )
                ious[i, j] = intersection / union

    return 1 - ious
