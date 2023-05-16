import numpy as np

from evaldet.dist import iou_dist


def test_iou() -> None:
    boxes_a = np.array(
        [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3], [0, 0, 4, 4]], dtype=np.float32
    )
    boxes_b = np.array([[0, 0, 3, 3], [3, 3, 4, 4]], dtype=np.float32)

    dists = iou_dist(boxes_a, boxes_b)
    exp_result = 1 - np.array([[1 / 9, 0], [4 / 9, 0], [1, 0], [9 / 16, 1 / 31]])
    np.testing.assert_almost_equal(dists, exp_result)
