import pytest

from evaldet import Tracks, compute_mot_metrics


def test_error_empty_metrics():
    with pytest.raises(ValueError, match="You must select some metrics"):
        compute_mot_metrics(
            Tracks(),
            Tracks(),
            clearmot_metrics=False,
            hota_metrics=False,
            id_metrics=False,
        )


def test_error_empty_ground_truth():
    with pytest.raises(ValueError, match=r"No objects in ``ground_truths``"):
        compute_mot_metrics(Tracks(), Tracks())
