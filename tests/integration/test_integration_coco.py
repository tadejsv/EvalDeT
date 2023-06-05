from pathlib import Path

import pytest

from evaldet.det.coco import compute_coco_summary
from evaldet.detections import Detections


def test_coco_integration_coco(data_dir: Path) -> None:
    hyp = Detections.from_coco(data_dir / "integration" / "dets_coco.json")
    gts = Detections.from_coco(data_dir / "integration" / "gts_coco.json")

    summary = compute_coco_summary(gts, hyp)

    assert summary["mean_ap"] == pytest.approx(0.503647, 1e-3)
    assert summary["ap_50"] == pytest.approx(0.696973, 1e-5)
    assert summary["ap_75"] == pytest.approx(0.571667, 1e-4)
    assert summary["mean_ap_sizes"]["small"] == pytest.approx(0.593252, 2e-3)
    assert summary["mean_ap_sizes"]["medium"] == pytest.approx(0.557991, 1e-4)
    assert summary["mean_ap_sizes"]["large"] == pytest.approx(0.489363, 1e-5)
