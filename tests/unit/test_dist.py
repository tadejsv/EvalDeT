from typing import Tuple

import numpy as np
import pytest

from evaldet.dist import iou_dist, iou_dist_pairwise


@pytest.fixture
def sample_inputs() -> Tuple[np.ndarray, np.ndarray]:
    inputs_1 = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    inputs_2 = np.array(
        [
            [0, 0, 1, 1],
            [10, 10, 11, 11],
            [-11, -11, -10, -10],
            [0.5, 0.5, 1, 1],
            [0.4, -1, 2, 0.3],
        ]
    )

    return inputs_1, inputs_2


def test_pairwise_distance(sample_inputs: Tuple[np.ndarray, np.ndarray]):
    inputs_1, inputs_2 = sample_inputs
    exp_results = np.array([0, 1, 1, 0.75, 0.937931])

    iou1 = iou_dist_pairwise(inputs_1, inputs_2)
    iou2 = iou_dist_pairwise(inputs_2, inputs_1)

    np.testing.assert_array_almost_equal(exp_results, iou1)
    np.testing.assert_array_almost_equal(exp_results, iou2)


def test_matrix_dist(sample_inputs: Tuple[np.ndarray, np.ndarray]):
    inputs_1, inputs_2 = sample_inputs
    line_results = np.array([0, 1, 1, 0.75, 0.937931])
    exp_results1 = np.tile(line_results, (inputs_1.shape[0], 1))
    exp_results2 = exp_results1.T

    iou1 = iou_dist(inputs_1, inputs_2)
    iou2 = iou_dist(inputs_2, inputs_1)

    np.testing.assert_array_almost_equal(exp_results1, iou1)
    np.testing.assert_array_almost_equal(exp_results2, iou2)


def test_matrix_dist2():
    inputs_1 = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]])
    inputs_2 = np.array([[3, 3, 13, 13], [4, 4, 14, 14]])

    exp_results = np.array(
        [
            [0.6754966887417219, 0.7804878048780488],
            [0.5294117647058824, 0.6754966887417219],
            [0.31932773109243695, 0.5294117647058824],
        ]
    )

    iou1 = iou_dist(inputs_1, inputs_2)
    iou2 = iou_dist(inputs_2, inputs_1)

    np.testing.assert_array_almost_equal(iou1, exp_results)
    np.testing.assert_array_almost_equal(iou2, exp_results.T)
