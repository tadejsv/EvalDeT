import pytest

from evaldet import MOTMetrics, Tracks


def test_error_empty_metrics() -> None:
    m = MOTMetrics()
    with pytest.raises(ValueError, match="You must select some metrics"):
        m.compute(
            Tracks([], [], []),
            Tracks([], [], []),
            clearmot_metrics=False,
            hota_metrics=False,
            id_metrics=False,
        )


def test_error_empty_ground_truth() -> None:
    m = MOTMetrics()
    with pytest.raises(ValueError, match=r"No objects in ``ground_truths``"):
        m.compute(Tracks([], [], []), Tracks([], [], []))
