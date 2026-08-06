"""Pose model management: specs, integrity, and the no-surprise-download rule.

Nothing here touches the network. The download path is exercised against a
local file:// URL, which covers the streaming, atomicity, and manifest logic
without depending on Google's servers or a working internet connection.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from golf_lab.pose.model_manager import (
    DEFAULT_MODEL_KEY,
    POSE_MODELS,
    PoseModelError,
    available_specs,
    download_model,
    ensure_model,
    get_spec,
    is_downloaded,
    manifest_entry,
    model_path,
    read_manifest,
    sha256_of,
    verify_model,
)


class TestSpecs:
    def test_default_model_exists(self):
        assert DEFAULT_MODEL_KEY in POSE_MODELS
        assert get_spec().key == DEFAULT_MODEL_KEY

    def test_unknown_key_is_rejected_with_the_options(self):
        with pytest.raises(PoseModelError) as info:
            get_spec("enormous")
        assert "lite" in str(info.value)

    def test_specs_are_ordered_by_cost(self):
        sizes = [spec.approx_megabytes for spec in available_specs()]
        assert sizes == sorted(sizes)

    def test_every_spec_records_its_licence_and_source(self):
        # A downloaded model is third-party content; where it came from and
        # under what licence has to be recorded, not assumed.
        for spec in POSE_MODELS.values():
            assert spec.license_name
            assert spec.license_url.startswith("https://")
            assert spec.source
            assert spec.url.startswith("https://")
            assert spec.filename.endswith(".task")


class TestNoSurpriseDownloads:
    def test_ensure_model_refuses_to_download_by_default(self, tmp_path):
        # Opening a page must never silently start a network transfer.
        with pytest.raises(PoseModelError) as info:
            ensure_model(get_spec("lite"), models_dir=tmp_path)

        message = str(info.value)
        assert "not downloaded yet" in message
        assert "Download button" in message
        assert not list(tmp_path.glob("*.task"))

    def test_ensure_model_returns_an_existing_file_without_network(self, tmp_path):
        spec = get_spec("lite")
        path = model_path(spec, tmp_path)
        path.write_bytes(b"pretend model")
        assert ensure_model(spec, models_dir=tmp_path) == path


class TestDownload:
    @pytest.fixture()
    def local_model_source(self, tmp_path):
        """A fake model served over file:// so no network is involved."""
        source = tmp_path / "source" / "pose_landmarker_lite.task"
        source.parent.mkdir(parents=True, exist_ok=True)
        payload = b"synthetic model bytes" * 500
        source.write_bytes(payload)
        return source, payload

    def _spec_pointing_at(self, source):
        import dataclasses

        return dataclasses.replace(get_spec("lite"), url=source.as_uri())

    def test_downloads_and_records_provenance(self, tmp_path, local_model_source):
        source, payload = local_model_source
        models_dir = tmp_path / "models"
        spec = self._spec_pointing_at(source)

        path = download_model(spec, models_dir=models_dir)

        assert path.read_bytes() == payload
        assert is_downloaded(spec, models_dir)

        entry = manifest_entry(spec, models_dir)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["size_bytes"] == len(payload)
        assert entry["license"] == spec.license_name
        assert entry["url"] == spec.url
        # Honest about what the checksum does and does not prove.
        assert entry["integrity"] == "trust-on-first-use"

    def test_reports_progress(self, tmp_path, local_model_source):
        source, _ = local_model_source
        updates = []

        download_model(
            self._spec_pointing_at(source),
            models_dir=tmp_path / "models",
            progress=lambda fraction, message: updates.append(fraction),
        )

        assert updates
        assert updates[-1] == pytest.approx(1.0)

    def test_leaves_no_partial_file_behind(self, tmp_path, local_model_source):
        source, _ = local_model_source
        models_dir = tmp_path / "models"
        download_model(self._spec_pointing_at(source), models_dir=models_dir)
        assert not list(models_dir.glob("*.part"))

    def test_existing_file_is_not_redownloaded(self, tmp_path, local_model_source):
        source, _ = local_model_source
        models_dir = tmp_path / "models"
        spec = self._spec_pointing_at(source)

        path = model_path(spec, models_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"already here")

        assert download_model(spec, models_dir=models_dir).read_bytes() == b"already here"

    def test_force_redownloads(self, tmp_path, local_model_source):
        source, payload = local_model_source
        models_dir = tmp_path / "models"
        spec = self._spec_pointing_at(source)

        path = model_path(spec, models_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

        assert download_model(spec, models_dir=models_dir, force=True).read_bytes() == payload

    def test_unreachable_source_gives_an_actionable_message(self, tmp_path):
        import dataclasses

        spec = dataclasses.replace(
            get_spec("lite"), url=(tmp_path / "missing.task").as_uri()
        )
        with pytest.raises(PoseModelError) as info:
            download_model(spec, models_dir=tmp_path / "models")

        assert "internet" in str(info.value).lower() or "reach" in str(info.value).lower()
        assert not list((tmp_path / "models").glob("*.part"))


class TestVerification:
    def test_missing_model_is_reported(self, tmp_path):
        assert "not been downloaded" in verify_model(get_spec("lite"), tmp_path)

    def test_intact_model_verifies_clean(self, tmp_path, monkeypatch):
        spec = get_spec("lite")
        path = model_path(spec, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model contents")

        (tmp_path / "pose_models.json").write_text(
            json.dumps({spec.key: {"sha256": sha256_of(path)}}), encoding="utf-8"
        )
        assert verify_model(spec, tmp_path) is None

    def test_corrupted_model_is_detected(self, tmp_path):
        spec = get_spec("lite")
        path = model_path(spec, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model contents")
        (tmp_path / "pose_models.json").write_text(
            json.dumps({spec.key: {"sha256": sha256_of(path)}}), encoding="utf-8"
        )

        path.write_bytes(b"truncated")  # simulate an interrupted download

        message = verify_model(spec, tmp_path)
        assert message is not None
        assert "corrupt" in message or "truncated" in message

    def test_model_without_a_recorded_checksum_is_flagged(self, tmp_path):
        spec = get_spec("lite")
        path = model_path(spec, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model contents")

        assert "cannot be confirmed" in verify_model(spec, tmp_path)

    def test_unreadable_manifest_does_not_crash(self, tmp_path):
        (tmp_path / "pose_models.json").write_text("{not json", encoding="utf-8")
        assert read_manifest(tmp_path) == {}
