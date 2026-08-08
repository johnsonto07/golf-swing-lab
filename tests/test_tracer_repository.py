"""Tracer storage: round-trips, isolation from derived artifacts, staleness."""

from __future__ import annotations

import json

from golf_lab.models.video import ShotShape
from golf_lab.storage import tracer_repository as repo
from golf_lab.storage.file_repository import swing_dir
from golf_lab.tracer.model import (
    TRACER_SCHEMA_VERSION,
    BallPoint,
    ImpactSource,
    PointSource,
    TracerHeight,
    TracerSpec,
)

SWING_ID = "20260101_120000_abcd1234"


def a_saved_spec(root, **kwargs) -> TracerSpec:
    swing_dir(SWING_ID, root).mkdir(parents=True, exist_ok=True)
    spec = TracerSpec(swing_id=SWING_ID, **kwargs)
    spec.confirm_impact(10, ImpactSource.USER)
    spec.add_point(BallPoint(frame=10, x=0.5, y=0.8, source=PointSource.CONFIRMED))
    spec.add_point(
        BallPoint(frame=16, x=0.6, y=0.3, source=PointSource.TRACKED, confidence=0.7)
    )
    return spec


class TestRoundTrip:
    def test_save_then_load_is_lossless(self, swing_root):
        spec = a_saved_spec(swing_root)
        spec.apply_preset(ShotShape.FADE, TracerHeight.HIGH)
        repo.save_tracer(spec, swing_root)

        loaded = repo.load_tracer(SWING_ID, swing_root)

        assert loaded is not None
        assert loaded.to_dict() == spec.to_dict()

    def test_provenance_survives_storage(self, swing_root):
        repo.save_tracer(a_saved_spec(swing_root), swing_root)

        loaded = repo.load_tracer(SWING_ID, swing_root)

        assert [p.source for p in loaded.points] == [
            PointSource.CONFIRMED,
            PointSource.TRACKED,
        ]
        assert loaded.points[1].confidence == 0.7

    def test_load_returns_none_when_absent(self, swing_root):
        assert repo.load_tracer(SWING_ID, swing_root) is None

    def test_has_tracer_reflects_the_file(self, swing_root):
        assert not repo.has_tracer(SWING_ID, swing_root)
        repo.save_tracer(a_saved_spec(swing_root), swing_root)
        assert repo.has_tracer(SWING_ID, swing_root)

    def test_written_as_readable_json(self, swing_root):
        repo.save_tracer(a_saved_spec(swing_root), swing_root)

        data = json.loads(
            repo.tracer_path(SWING_ID, swing_root).read_text(encoding="utf-8")
        )

        assert data["impact_frame"] == 10
        assert data["impact_source"] == "user"
        assert len(data["points"]) == 2

    def test_saving_leaves_no_temp_file(self, swing_root):
        repo.save_tracer(a_saved_spec(swing_root), swing_root)

        leftovers = list(swing_dir(SWING_ID, swing_root).glob("*.tmp"))
        assert leftovers == []

    def test_unreadable_tracer_returns_none_rather_than_raising(self, swing_root):
        swing_dir(SWING_ID, swing_root).mkdir(parents=True, exist_ok=True)
        repo.tracer_path(SWING_ID, swing_root).write_text("{not json", encoding="utf-8")

        assert repo.load_tracer(SWING_ID, swing_root) is None

    def test_resaving_overwrites_rather_than_appends(self, swing_root):
        spec = a_saved_spec(swing_root)
        repo.save_tracer(spec, swing_root)

        spec.remove_point(16)
        repo.save_tracer(spec, swing_root)

        loaded = repo.load_tracer(SWING_ID, swing_root)
        assert [p.frame for p in loaded.points] == [10]


class TestIsolation:
    def test_tracer_lives_beside_other_artifacts_not_inside_them(self, swing_root):
        repo.save_tracer(a_saved_spec(swing_root), swing_root)

        path = repo.tracer_path(SWING_ID, swing_root)
        assert path.name == "tracer.json"
        assert path.parent == swing_dir(SWING_ID, swing_root)

    def test_deleting_the_tracer_touches_nothing_else(self, swing_root):
        directory = swing_dir(SWING_ID, swing_root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "swing_analysis.json").write_text("{}", encoding="utf-8")
        repo.save_tracer(a_saved_spec(swing_root), swing_root)

        repo.delete_tracer(SWING_ID, swing_root)

        assert not repo.has_tracer(SWING_ID, swing_root)
        assert (directory / "swing_analysis.json").exists()

    def test_deleting_a_missing_tracer_is_quiet(self, swing_root):
        repo.delete_tracer(SWING_ID, swing_root)  # must not raise


class TestStaleness:
    def test_a_fresh_tracer_is_not_stale(self, swing_root):
        spec = a_saved_spec(swing_root, preview_fingerprint="abc123")
        assert repo.staleness_reasons(spec, "abc123") == []
        assert not repo.is_stale(spec, "abc123")

    def test_regenerated_preview_is_reported(self, swing_root):
        spec = a_saved_spec(swing_root, preview_fingerprint="abc123")

        reasons = repo.staleness_reasons(spec, "different")

        assert len(reasons) == 1
        assert "regenerated" in reasons[0]

    def test_schema_change_is_reported(self, swing_root):
        spec = a_saved_spec(swing_root)
        spec.schema_version = TRACER_SCHEMA_VERSION + 1

        assert any("schema" in r for r in repo.staleness_reasons(spec))

    def test_absent_tracer_is_not_stale(self):
        # Nothing drawn yet is not the same as something out of date.
        assert repo.staleness_reasons(None) == []

    def test_unknown_fingerprint_is_not_treated_as_a_mismatch(self, swing_root):
        spec = a_saved_spec(swing_root, preview_fingerprint="abc123")
        assert repo.staleness_reasons(spec, "") == []

    def test_a_stale_tracer_is_still_loadable(self, swing_root):
        # Hand-drawn work is reported stale, never discarded.
        spec = a_saved_spec(swing_root, preview_fingerprint="abc123")
        repo.save_tracer(spec, swing_root)

        loaded = repo.load_tracer(SWING_ID, swing_root)

        assert repo.is_stale(loaded, "different")
        assert len(loaded.points) == 2
