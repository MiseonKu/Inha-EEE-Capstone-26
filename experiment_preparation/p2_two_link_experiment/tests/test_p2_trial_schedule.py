from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p2_forward_targets import generate_targets
from sampling.p2_trial_schedule import generate_schedule


def test_schedule_repeats_and_order() -> None:
    targets, _ = generate_targets(
        ROOT / "config" / "p2_geometry.yaml",
        ROOT / "config" / "p2_forward_workspace.yaml",
        target_count=18,
        candidate_count=900,
        seed=11,
    )
    schedule, metadata = generate_schedule(targets, repeats=3, seed=12)
    assert schedule["trial_id"].is_unique
    assert schedule.groupby("target_id").size().eq(3).all()
    assert metadata["all_targets_expected_repeats"]
    assert metadata["adjacent_same_target_count"] == 0
    assert len(schedule) == 54
