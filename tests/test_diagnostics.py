"""Diagnostics must always produce a report and never leak secrets."""

from __future__ import annotations

from golf_lab.config import openai_integration_enabled
from golf_lab.diagnostics import collect_diagnostics
from golf_lab.video.frame_cache import FrameCache, frame_cache_key


class TestDiagnostics:
    def test_collects_without_raising(self):
        report = collect_diagnostics()
        assert report.python_version
        assert report.app_version
        assert "streamlit" in report.packages

    def test_text_output_is_renderable(self):
        text = collect_diagnostics().to_text()
        assert "Golf Swing Lab diagnostics" in text
        assert "Inference device" in text

    def test_never_prints_the_api_key(self, monkeypatch):
        secret = "sk-thisisafakekeyvalue1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", secret)

        report = collect_diagnostics()
        assert report.openai_enabled is True
        assert secret not in report.to_text()

    def test_reports_disabled_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_integration_enabled() is False
        assert collect_diagnostics().openai_enabled is False

    def test_gpu_absence_is_not_an_error(self):
        gpu = collect_diagnostics().gpu
        assert "summary" in gpu
        assert isinstance(gpu.get("available", False), bool)


class TestFrameCache:
    def test_keys_are_deterministic(self, fixture_video):
        a = frame_cache_key(fixture_video, 12, kind="bgr")
        b = frame_cache_key(fixture_video, 12, kind="bgr")
        assert a == b

    def test_keys_differ_by_frame_and_settings(self, fixture_video):
        base = frame_cache_key(fixture_video, 12, kind="bgr")
        assert base != frame_cache_key(fixture_video, 13, kind="bgr")
        assert base != frame_cache_key(fixture_video, 12, kind="rgb")

    def test_evicts_least_recently_used(self):
        cache = FrameCache(max_items=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")          # 'a' is now the most recently used
        cache.put("c", 3)       # evicts 'b'
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_respects_max_size(self):
        cache = FrameCache(max_items=3)
        for index in range(10):
            cache.put(str(index), index)
        assert len(cache) == 3
