import pytest

from evaldet import Tracks, compute_mot_metrics


def test_error_not_allowed_metrics():
    with pytest.raises(ValueError, match=r"These .* \[\'wrong\'\]"):
        compute_mot_metrics(["wrong", "MOTA"], Tracks(), Tracks())


def test_error_empty_metrics():
    with pytest.raises(ValueError, match=r"The ``metrics`` sequence is empty"):
        compute_mot_metrics([], Tracks(), Tracks())


def test_error_empty_ground_truth():
    with pytest.raises(ValueError, match=r"No objects in ``ground_truths``"):
        compute_mot_metrics(["MOTA"], Tracks(), Tracks())
