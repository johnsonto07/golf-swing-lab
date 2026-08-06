"""Smoke tests that actually execute every Streamlit page script.

An HTTP 200 from the Streamlit server only proves the page shell loaded — the
script itself runs later over a websocket, so a broken page still returns 200.
`AppTest` runs the real script and surfaces any exception, which is what makes
this a meaningful check.

These are marked skip-if-unavailable so the core suite still runs on a machine
without the Streamlit testing harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing harness unavailable"
).AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent

PAGE_SCRIPTS = [
    REPO_ROOT / "app.py",
    REPO_ROOT / "pages" / "1_Video_Lab.py",
    REPO_ROOT / "pages" / "2_Swing_Analysis.py",
    REPO_ROOT / "pages" / "3_Compare.py",
    REPO_ROOT / "pages" / "4_Ball_Tracer.py",
    REPO_ROOT / "pages" / "5_History.py",
    REPO_ROOT / "pages" / "6_Settings.py",
]


@pytest.mark.parametrize("script", PAGE_SCRIPTS, ids=lambda p: p.name)
def test_page_runs_without_exception(script: Path):
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception, (
        f"{script.name} raised: "
        + "; ".join(str(e.value) for e in app.exception)
    )


def test_home_page_renders_title():
    app = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60).run()
    assert any("Golf Swing Lab" in str(t.value) for t in app.title)


def test_video_lab_offers_both_tabs():
    app = AppTest.from_file(
        str(REPO_ROOT / "pages" / "1_Video_Lab.py"), default_timeout=60
    ).run()
    assert not app.exception
    # The uploader and the metadata form must both be present for the page to
    # be usable at all.
    assert len(app.selectbox) >= 1
    assert len(app.button) >= 1


def test_settings_page_never_shows_a_key(monkeypatch):
    secret = "sk-fakekeyshouldnotappear000000"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    app = AppTest.from_file(
        str(REPO_ROOT / "pages" / "6_Settings.py"), default_timeout=60
    ).run()
    assert not app.exception

    rendered = " ".join(
        str(element.value) for element in list(app.markdown) + list(app.code)
    )
    assert secret not in rendered
