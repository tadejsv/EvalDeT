import numba
import numpy as np
import numpy.typing as npt


@numba.njit(
    numba.int32[:, ::1](
        numba.int32[::1],
        numba.boolean[::1],
        numba.int32[::1],
        numba.int32[::1],
        numba.int32,
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
    Compute the confusion matrix

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

        if gt_ind > -1 and gt_ignored[gt_ind] == True:
            c = -1
        elif gt_ind > -1:
            c = gt_classes[gt_ind]

            # Subtract match from gt totals
            conf_matrix[-1, c] -= 1
        else:
            c = -1

        conf_matrix[r, c] += 1

    return conf_matrix
