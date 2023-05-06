import numpy as np

from evaldet import MOTMetrics, Tracks


def test_hyp_missing_frame(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    m = MOTMetrics()

    gt, hyp = missing_frame_pair
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    assert metrics["hota"]["DetA"] == 0.5
    assert metrics["hota"]["AssA"] == 0.5
    assert metrics["hota"]["HOTA"] == 0.5
    assert metrics["hota"]["LocA"] == 1.0
    for metric in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics["hota"][metric].var() == 0  # type: ignore


def test_gt_missing_frame(missing_frame_pair: tuple[Tracks, Tracks]) -> None:
    m = MOTMetrics()

    gt, hyp = missing_frame_pair
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    assert metrics["hota"]["DetA"] == 0.5
    assert metrics["hota"]["AssA"] == 0.5
    assert metrics["hota"]["HOTA"] == 0.5
    assert metrics["hota"]["LocA"] == 1.0
    for metric in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics["hota"][metric].var() == 0  # type: ignore


def test_no_matches() -> None:
    m = MOTMetrics()

    gt = Tracks([0], [0], np.array([[0, 0, 1, 1]]))
    hyp = Tracks([0], [0], np.array([[1, 1, 1, 1]]))
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    assert metrics["hota"]["DetA"] == 0
    assert metrics["hota"]["AssA"] == 0
    assert metrics["hota"]["HOTA"] == 0
    assert metrics["hota"]["LocA"] == 1.0
    for metric in ["HOTA_alpha", "AssA_alpha", "DetA_alpha", "LocA_alpha"]:
        assert metrics["hota"][metric].var() == 0  # type: ignore


def test_alphas() -> None:
    m = MOTMetrics()

    gt = Tracks([0, 1], [0, 0], np.array([[0, 0, 1, 1], [1, 1, 1, 1]]))
    hyp = Tracks([0, 1], [0, 0], np.array([[0.1, 0.1, 1, 1], [1.2, 1.2, 1, 1]]))
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    np.testing.assert_almost_equal(metrics["hota"]["AssA"], 0.6842105263157, 5)
    np.testing.assert_almost_equal(metrics["hota"]["DetA"], 0.5438596491228, 5)
    np.testing.assert_almost_equal(metrics["hota"]["HOTA"], 0.5952316356188, 5)
    np.testing.assert_almost_equal(metrics["hota"]["LocA"], 0.7317558602388, 5)

    for a_ind, alpha in enumerate(metrics["hota"]["alphas_HOTA"]):
        if alpha < 0.31932773:
            assert metrics["hota"]["HOTA_alpha"][a_ind] == 0
            assert metrics["hota"]["AssA_alpha"][a_ind] == 0
            assert metrics["hota"]["DetA_alpha"][a_ind] == 0
            assert metrics["hota"]["LocA_alpha"][a_ind] == 1.0
        elif 0.52941176 > alpha >= 0.31932773:
            assert metrics["hota"]["HOTA_alpha"][a_ind] == np.sqrt(1 / 3)
            assert metrics["hota"]["AssA_alpha"][a_ind] == 1.0
            assert metrics["hota"]["DetA_alpha"][a_ind] == 1 / 3
            np.testing.assert_almost_equal(
                metrics["hota"]["LocA_alpha"][a_ind], 1 - 0.31932773, 5
            )
        else:
            assert metrics["hota"]["HOTA_alpha"][a_ind] == 1
            assert metrics["hota"]["AssA_alpha"][a_ind] == 1.0
            assert metrics["hota"]["DetA_alpha"][a_ind] == 1
            np.testing.assert_almost_equal(
                metrics["hota"]["LocA_alpha"][a_ind],
                1 - (0.31932773 + 0.52941176) / 2,
                5,
            )


def test_priority_matching_1() -> None:
    """Test that priority is making TPs, when possible (and take association
    into account only AFTER that).

    Inspired from https://github.com/JonathonLuiten/TrackEval/issues/22
    """
    m = MOTMetrics()

    gt = Tracks(
        ids=[0] * 10 + [0, 1],
        frame_nums=list(range(10)) + [10, 10],
        detections=np.array(
            [[0, 0, 1, 1]] * 10 + [[0, 0, 1, 1 + 1e-5], [0, 0, 1, 1 - 1e-5]]
        ),
    )

    # This is set up in a way that dist from 0 will be just above 0.5,
    # and distance from 1 below 0.5
    hyp = Tracks(
        ids=[0] * 10 + [0],
        frame_nums=list(range(10)) + [10],
        detections=np.array([[0, 0, 1, 1]] * 10 + [[0, 0, 1, 0.5]]),
    )

    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    np.testing.assert_array_almost_equal(metrics["hota"]["AssA_alpha"][9], 0.7658, 3)
    np.testing.assert_array_almost_equal(metrics["hota"]["DetA_alpha"][9], 11 / 12, 3)
    np.testing.assert_array_almost_equal(metrics["hota"]["HOTA_alpha"][9], 0.8378, 3)


def test_priority_matching_2() -> None:
    """Test that when we have two equally viable matches, the preference is
    taken by what has higher A(c), and not what has higher similarity."""
    m = MOTMetrics()

    gt = Tracks(
        [0, 0, 1], [0, 1, 1], np.array([[0, 0, 1, 1], [0, 0, 1, 0.7], [0, 0, 1, 1]])
    )
    hyp = Tracks([0, 0], [0, 1], np.array([[0, 0, 1, 1]] * 2))
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    assert metrics["hota"]["AssA_alpha"][9] == 1
    assert metrics["hota"]["DetA_alpha"][9] == 2 / 3
    np.testing.assert_array_almost_equal(metrics["hota"]["HOTA_alpha"][9], 0.8164, 3)


def test_example_1a() -> None:
    """Example A from figure 1 in the paper"""
    m = MOTMetrics()

    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks([0] * 10, list(range(10)), np.array([[0, 0, 1, 1]] * 10))
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    assert metrics["hota"]["HOTA"] == 0.5
    assert metrics["hota"]["AssA"] == 0.5
    assert metrics["hota"]["DetA"] == 0.5
    assert metrics["hota"]["LocA"] == 1.0


def test_example_1b() -> None:
    """Example B from figure 1 in the paper"""
    m = MOTMetrics()

    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks([0] * 7 + [1] * 7, list(range(14)), np.array([[0, 0, 1, 1]] * 14))
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    np.testing.assert_array_almost_equal(metrics["hota"]["HOTA"], 0.5, 2)
    np.testing.assert_array_almost_equal(metrics["hota"]["AssA"], 0.35, 2)
    np.testing.assert_array_almost_equal(metrics["hota"]["DetA"], 0.70, 2)
    assert metrics["hota"]["LocA"] == 1.0


def test_example_1c() -> None:
    """Example C from figure 1 in the paper"""
    m = MOTMetrics()

    gt = Tracks([0] * 20, list(range(20)), np.array([[0, 0, 1, 1]] * 20))
    hyp = Tracks(
        [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
        list(range(20)),
        np.array([[0, 0, 1, 1]] * 20),
    )
    metrics = m.compute(
        gt, hyp, hota_metrics=True, clearmot_metrics=False, id_metrics=False
    )

    assert metrics["hota"] is not None
    np.testing.assert_array_almost_equal(metrics["hota"]["HOTA"], 0.5, 2)
    np.testing.assert_array_almost_equal(metrics["hota"]["AssA"], 0.25, 2)
    np.testing.assert_array_almost_equal(metrics["hota"]["DetA"], 1, 2)
    assert metrics["hota"]["LocA"] == 1.0
