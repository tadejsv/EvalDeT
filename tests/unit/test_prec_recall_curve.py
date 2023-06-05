import numpy as np
import numpy.testing as npt

from evaldet.utils.prec_recall import prec_recall_curve as prc


def test_prec_recall_curve_no_gts() -> None:
    assert prc(np.array([False]), np.array([1.0]), 0) == (None, None)


def test_prec_recall_curve_no_preds() -> None:
    prec, rec = prc(np.array([]), np.array([]), 1)
    assert prec is not None and rec is not None
    npt.assert_array_equal(prec, np.array([0.0]))
    npt.assert_array_equal(rec, np.array([0.0]))


def test_prec_recall_curve_no_matching() -> None:
    prec, rec = prc(np.array([False, False]), np.array([1.0, 1.0]), 2)
    assert prec is not None and rec is not None
    npt.assert_array_equal(prec, np.array([0.0, 0.0]))
    npt.assert_array_equal(rec, np.array([0.0, 0.0]))


def test_prec_recall_curve_normal() -> None:
    det_matched = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=np.bool_,
    )
    det_conf = np.array(
        [
            0.95,
            0.98,
            0.951,
            0.99,
            0.94,
            0.92,
            0.89,
            0.86,
            0.85,
            0.82,
            0.81,
            0.76,
        ],
        dtype=np.float32,
    )
    prec, rec = prc(det_matched, det_conf, 12)
    assert prec is not None and rec is not None
    npt.assert_almost_equal(
        prec,
        np.array(
            [
                1.0,
                0.5,
                0.66666667,
                0.5,
                0.6,
                0.66666667,
                0.71428571,
                0.75,
                0.66666667,
                0.7,
                0.63636364,
                0.66666667,
            ]
        ),
        decimal=4,
    )
    npt.assert_almost_equal(
        rec,
        np.array(
            [
                0.08333333,
                0.08333333,
                0.16666667,
                0.16666667,
                0.25,
                0.33333333,
                0.41666667,
                0.5,
                0.5,
                0.58333333,
                0.58333333,
                0.66666667,
            ]
        ),
        decimal=4,
    )
