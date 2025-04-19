import numpy as np
import pytest

from evaldet.mot.hota import calculate_hota_metrics
from evaldet.mot.motmetrics import _compute_ious
from evaldet.tracks import Tracks


def test_hyp_missing_frame(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    gt, hyp = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["DetA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["HOTA"] == 0.5
    assert metrics["LocA"] == 1.0
    for metric in [
        "HOTA_alpha",
        "AssA_alpha",
        "DetA_alpha",
        "LocA_alpha",
        "AssRec_alpha",
        "AssPr_alpha",
        "DetPr_alpha",
        "DetRec_alpha",
        "DetFN_alpha",
        "DetFP_alpha",
        "DetTP_alpha",
    ]:
        assert pytest.approx(metrics[metric].var()) == 0  # type: ignore[literal-required]


def test_gt_missing_frame(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    gt, hyp = missing_frame_pair
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["DetA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["HOTA"] == 0.5
    assert metrics["LocA"] == 1.0
    for metric in [
        "HOTA_alpha",
        "AssA_alpha",
        "DetA_alpha",
        "LocA_alpha",
        "AssRec_alpha",
        "AssPr_alpha",
        "DetPr_alpha",
        "DetRec_alpha",
        "DetFN_alpha",
        "DetFP_alpha",
        "DetTP_alpha",
    ]:
        assert pytest.approx(metrics[metric].var()) == 0  # type: ignore[literal-required]


def test_no_matches() -> None:
    gt = Tracks([0], [0], np.array([[0, 0, 1, 1]]))
    hyp = Tracks([0], [0], np.array([[1, 1, 1, 1]]))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["DetA"] == 0
    assert metrics["AssA"] == 0
    assert metrics["HOTA"] == 0
    assert metrics["LocA"] == 1.0
    for metric in [
        "HOTA_alpha",
        "AssA_alpha",
        "DetA_alpha",
        "LocA_alpha",
        "AssRec_alpha",
        "AssPr_alpha",
        "DetPr_alpha",
        "DetRec_alpha",
        "DetFN_alpha",
        "DetFP_alpha",
        "DetTP_alpha",
    ]:
        assert pytest.approx(metrics[metric].var()) == 0  # type: ignore[literal-required]


def test_alphas() -> None:
    gt = Tracks([0, 1], [0, 0], np.array([[0, 0, 1, 1], [1, 1, 1, 1]]))
    hyp = Tracks([0, 1], [0, 0], np.array([[0.1, 0.1, 1, 1], [1.2, 1.2, 1, 1]]))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["AssA"] == pytest.approx(0.6842105263157, 1e-5)
    assert metrics["DetA"] == pytest.approx(0.5438596491228, 1e-5)
    assert metrics["HOTA"] == pytest.approx(0.5952316356188, 1e-5)
    assert metrics["LocA"] == pytest.approx(0.7317558602388, 1e-5)

    for a_ind, alpha in enumerate(metrics["alphas_HOTA"]):
        if alpha < 0.31932773:
            assert metrics["HOTA_alpha"][a_ind] == 0
            assert metrics["AssA_alpha"][a_ind] == 0
            assert metrics["DetA_alpha"][a_ind] == 0
            assert metrics["AssPr_alpha"][a_ind] == 0
            assert metrics["AssRec_alpha"][a_ind] == 0
            assert metrics["DetPr_alpha"][a_ind] == 0
            assert metrics["DetRec_alpha"][a_ind] == 0
            assert metrics["DetTP_alpha"][a_ind] == 0
            assert metrics["DetFP_alpha"][a_ind] == 2
            assert metrics["DetFN_alpha"][a_ind] == 2

            assert metrics["LocA_alpha"][a_ind] == 1.0
        elif 0.52941176 > alpha >= 0.31932773:
            assert metrics["HOTA_alpha"][a_ind] == np.sqrt(1 / 3)
            assert metrics["AssA_alpha"][a_ind] == 1.0
            assert metrics["DetA_alpha"][a_ind] == 1 / 3

            assert metrics["AssPr_alpha"][a_ind] == 1
            assert metrics["AssRec_alpha"][a_ind] == 1
            assert metrics["DetPr_alpha"][a_ind] == 0.5
            assert metrics["DetRec_alpha"][a_ind] == 0.5
            assert metrics["DetTP_alpha"][a_ind] == 1
            assert metrics["DetFP_alpha"][a_ind] == 1
            assert metrics["DetFN_alpha"][a_ind] == 1

            assert metrics["LocA_alpha"][a_ind] == pytest.approx(1 - 0.31932773, 1e-5)
        else:
            assert metrics["HOTA_alpha"][a_ind] == 1
            assert metrics["AssA_alpha"][a_ind] == 1.0
            assert metrics["DetA_alpha"][a_ind] == 1

            assert metrics["DetPr_alpha"][a_ind] == 1
            assert metrics["DetRec_alpha"][a_ind] == 1
            assert metrics["AssPr_alpha"][a_ind] == 1
            assert metrics["AssRec_alpha"][a_ind] == 1
            assert metrics["DetTP_alpha"][a_ind] == 2
            assert metrics["DetFP_alpha"][a_ind] == 0
            assert metrics["DetFN_alpha"][a_ind] == 0

            assert metrics["LocA_alpha"][a_ind] == pytest.approx(
                1 - (0.31932773 + 0.52941176) / 2, 1e-5
            )


def test_priority_matching_1() -> None:
    """Test that priority is making TPs, when possible (and take association
    into account only AFTER that).

    Inspired from https://github.com/JonathonLuiten/TrackEval/issues/22
    """

    gt = Tracks(
        ids=[0] * 10 + [0, 1],
        frame_nums=[*list(range(10)), 10, 10],
        bboxes=np.array(
            [[0, 0, 1, 1]] * 10 + [[0, 0, 1, 1 + 1e-5], [0, 0, 1, 1 - 1e-5]]
        ),
    )

    # This is set up in a way that dist from 0 will be just above 0.5,
    # and distance from 1 below 0.5
    hyp = Tracks(
        ids=[0] * 10 + [0],
        frame_nums=[*list(range(10)), 10],
        bboxes=np.array([[0, 0, 1, 1]] * 10 + [[0, 0, 1, 0.5]]),
    )
    (10 / 11) * (10 / 11)
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics["AssA_alpha"][9] == pytest.approx(0.7658, 1e-3)
    assert metrics["DetA_alpha"][9] == pytest.approx(11 / 12)
    assert metrics["HOTA_alpha"][9] == pytest.approx(0.8378, 1e-3)

    assert metrics["DetPr_alpha"][9] == pytest.approx(1)
    assert metrics["DetRec_alpha"][9] == pytest.approx(11 / 12)
    assert metrics["AssPr_alpha"][9] == pytest.approx((10 * 10 / 11 + 1 * 1 / 11) / 11)
    assert metrics["AssRec_alpha"][9] == pytest.approx((10 * 10 / 11 + 1 * 1 / 1) / 11)
    assert metrics["DetFN_alpha"][9] == 1
    assert metrics["DetTP_alpha"][9] == 11
    assert metrics["DetFP_alpha"][9] == 0


def test_priority_matching_2() -> None:
    """Test that when we have two equally viable matches, the preference is
    taken by what has higher A(c), and not what has higher similarity."""
    gt = Tracks(
        [0, 0, 1], [0, 1, 1], np.array([[0, 0, 1, 1], [0, 0, 1, 0.7], [0, 0, 1, 1]])
    )
    hyp = Tracks([0, 0], [0, 1], np.array([[0, 0, 1, 1]] * 2))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["AssA_alpha"][9] == 1
    assert metrics["DetA_alpha"][9] == 2 / 3
    assert metrics["HOTA_alpha"][9] == pytest.approx(0.8164, 1e-3)

    assert metrics["DetPr_alpha"][9] == 1
    assert metrics["DetRec_alpha"][9] == 2 / 3
    assert metrics["AssPr_alpha"][9] == 1
    assert metrics["AssRec_alpha"][9] == 1
    assert metrics["DetFN_alpha"][9] == 1
    assert metrics["DetTP_alpha"][9] == 2
    assert metrics["DetFP_alpha"][9] == 0


def test_example_1a() -> None:
    """Example A from figure 1 in the paper"""
    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks([0] * 10, list(range(10)), np.array([[0, 0, 1, 1]] * 10))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["HOTA"] == 0.5
    assert metrics["AssA"] == 0.5
    assert metrics["DetA"] == 0.5
    assert metrics["LocA"] == 1.0

    for a_ind in range(len(metrics["alphas_HOTA"])):
        assert metrics["HOTA_alpha"][a_ind] == 0.5
        assert metrics["AssA_alpha"][a_ind] == 0.5
        assert metrics["DetA_alpha"][a_ind] == 0.5

        assert metrics["DetPr_alpha"][9] == 1
        assert metrics["DetRec_alpha"][9] == 0.5
        assert metrics["AssPr_alpha"][9] == 1
        assert metrics["AssRec_alpha"][9] == 0.5
        assert metrics["DetFN_alpha"][9] == 10
        assert metrics["DetTP_alpha"][9] == 10
        assert metrics["DetFP_alpha"][9] == 0


def test_example_1b() -> None:
    """Example B from figure 1 in the paper"""
    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks([0] * 7 + [1] * 7, list(range(14)), np.array([[0, 0, 1, 1]] * 14))
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["HOTA"] == pytest.approx(np.sqrt(0.35 * 0.7))
    assert metrics["AssA"] == pytest.approx(0.35)
    assert metrics["DetA"] == pytest.approx(0.70)
    assert metrics["LocA"] == 1.0

    for a_ind in range(len(metrics["alphas_HOTA"])):
        assert metrics["HOTA_alpha"][a_ind] == pytest.approx(np.sqrt(0.35 * 0.7))
        assert metrics["AssA_alpha"][a_ind] == pytest.approx(0.35)
        assert metrics["DetA_alpha"][a_ind] == 0.70

        assert metrics["DetPr_alpha"][9] == 1
        assert metrics["DetRec_alpha"][9] == pytest.approx(0.7)
        assert metrics["AssPr_alpha"][9] == 1
        assert metrics["AssRec_alpha"][9] == pytest.approx(0.35)
        assert metrics["DetFN_alpha"][9] == 6
        assert metrics["DetTP_alpha"][9] == 14
        assert metrics["DetFP_alpha"][9] == 0


def test_example_1c() -> None:
    """Example C from figure 1 in the paper"""
    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks(
        [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
        list(range(20)),
        np.array([[0, 0, 1, 1]] * 20),
    )
    ious = _compute_ious(gt, hyp)
    metrics = calculate_hota_metrics(gt, hyp, ious=ious)

    assert metrics is not None
    assert metrics["HOTA"] == pytest.approx(0.5)
    assert metrics["AssA"] == pytest.approx(0.25)
    assert metrics["DetA"] == 1
    assert metrics["LocA"] == 1.0

    for a_ind in range(len(metrics["alphas_HOTA"])):
        assert metrics["HOTA_alpha"][a_ind] == pytest.approx(0.5)
        assert metrics["AssA_alpha"][a_ind] == pytest.approx(0.25)
        assert metrics["DetA_alpha"][a_ind] == 1

        assert metrics["DetPr_alpha"][9] == 1
        assert metrics["DetRec_alpha"][9] == 1
        assert metrics["AssPr_alpha"][9] == 1
        assert metrics["AssRec_alpha"][9] == pytest.approx(0.25)
        assert metrics["DetFN_alpha"][9] == 0
        assert metrics["DetTP_alpha"][9] == 20
        assert metrics["DetFP_alpha"][9] == 0
