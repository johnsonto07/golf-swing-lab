"""Pose model files: which one, where from, what licence, and is it intact.

The app ships no model weights. A ``.task`` bundle is fetched on explicit
user request into ``models/`` and never automatically: a download is the only
moment this otherwise-offline app touches the network, so it is a deliberate,
visible action rather than a side effect of opening a page.

Integrity handling is trust-on-first-use, and says so. Google does not publish
per-file checksums for these bundles, so inventing a pinned hash here would be
security theatre. Instead the hash of whatever was actually downloaded is
recorded in ``models/pose_models.json`` and verified on every later load, which
does catch the realistic failure — a truncated or corrupted file — even though
it cannot by itself prove the first download was authentic.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from golf_lab.config import MODELS_DIR
from golf_lab.logging_config import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "pose_models.json"
DOWNLOAD_TIMEOUT_SECONDS = 60
_CHUNK_BYTES = 1024 * 256

ProgressCallback = Callable[[float, str], None]


class PoseModelError(RuntimeError):
    """Raised for a missing, corrupt, or undownloadable model. User-facing."""


@dataclass(frozen=True)
class PoseModelSpec:
    """One selectable pose model."""

    key: str
    display_name: str
    filename: str
    url: str
    approx_megabytes: float
    description: str
    license_name: str = "Apache License 2.0"
    license_url: str = "https://www.apache.org/licenses/LICENSE-2.0"
    source: str = "Google MediaPipe Pose Landmarker"
    model_version: str = "1"


# float16 bundles: the accuracy/speed ladder MediaPipe publishes for Pose
# Landmarker. "full" is the default because on a golf swing the lite model
# loses the trail wrist at speed often enough to matter, and heavy costs
# several times the runtime on CPU for a smaller gain.
POSE_MODELS: Dict[str, PoseModelSpec] = {
    "lite": PoseModelSpec(
        key="lite",
        display_name="Lite — fastest, least accurate",
        filename="pose_landmarker_lite.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        ),
        approx_megabytes=5.5,
        description=(
            "Use when you just want a quick look, or on a slow machine. "
            "Tends to lose fast-moving wrists near the top of the backswing."
        ),
    ),
    "full": PoseModelSpec(
        key="full",
        display_name="Full — recommended balance",
        filename="pose_landmarker_full.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task"
        ),
        approx_megabytes=9.0,
        description=(
            "The default. Holds the arms and hips through a full-speed swing "
            "on ordinary phone footage without being painfully slow on CPU."
        ),
    ),
    "heavy": PoseModelSpec(
        key="heavy",
        display_name="Heavy — most accurate, slowest",
        filename="pose_landmarker_heavy.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        ),
        approx_megabytes=29.2,
        description=(
            "Worth it for a swing you care about analysing carefully. "
            "Expect roughly three to five times the CPU time of Full."
        ),
    ),
}

DEFAULT_MODEL_KEY = "full"


def get_spec(key: str = DEFAULT_MODEL_KEY) -> PoseModelSpec:
    try:
        return POSE_MODELS[key]
    except KeyError:
        raise PoseModelError(
            f"Unknown pose model '{key}'. Available: {', '.join(POSE_MODELS)}."
        ) from None


def available_specs() -> List[PoseModelSpec]:
    """Specs in ascending cost order, for presenting as a choice."""
    return [POSE_MODELS[key] for key in ("lite", "full", "heavy")]


def model_path(spec: PoseModelSpec, models_dir: Optional[Path] = None) -> Path:
    return Path(models_dir or MODELS_DIR) / spec.filename


def is_downloaded(spec: PoseModelSpec, models_dir: Optional[Path] = None) -> bool:
    path = model_path(spec, models_dir)
    return path.exists() and path.stat().st_size > 0


# --- manifest ------------------------------------------------------------
def _manifest_path(models_dir: Optional[Path] = None) -> Path:
    return Path(models_dir or MODELS_DIR) / MANIFEST_NAME


def read_manifest(models_dir: Optional[Path] = None) -> Dict[str, dict]:
    """Recorded provenance for every downloaded model.

    A corrupt manifest is treated as an empty one: it is a cache of things we
    can re-derive, and refusing to start because of it would be worse than
    re-verifying.
    """
    path = _manifest_path(models_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable pose model manifest %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(manifest: Dict[str, dict], models_dir: Optional[Path] = None) -> None:
    path = _manifest_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 so a 30 MB model is not loaded into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(
    spec: PoseModelSpec, models_dir: Optional[Path] = None
) -> Optional[dict]:
    return read_manifest(models_dir).get(spec.key)


# --- download ------------------------------------------------------------
def download_model(
    spec: PoseModelSpec,
    models_dir: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    force: bool = False,
) -> Path:
    """Fetch ``spec`` into ``models_dir`` and record its provenance.

    Downloads to a temporary file and only moves it into place once the whole
    body has arrived, so an interrupted download can never leave a
    half-written ``.task`` that later fails deep inside MediaPipe with an
    unreadable error.
    """
    report = progress or (lambda fraction, message: None)
    destination = model_path(spec, models_dir)

    if destination.exists() and not force:
        logger.info("Pose model already present: %s", destination.name)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose model %s from %s", spec.key, spec.url)
    report(0.0, f"Connecting to {spec.source}")

    temp_handle = tempfile.NamedTemporaryFile(
        delete=False, dir=str(destination.parent), suffix=".part"
    )
    temp_path = Path(temp_handle.name)
    try:
        with temp_handle:
            request = urllib.request.Request(
                spec.url, headers={"User-Agent": "golf-swing-lab"}
            )
            with urllib.request.urlopen(  # noqa: S310 - fixed https URL from a constant
                request, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    temp_handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        report(
                            downloaded / total,
                            f"Downloading {spec.display_name} "
                            f"({downloaded / 1e6:.1f} of {total / 1e6:.1f} MB)",
                        )
                    else:
                        report(0.0, f"Downloading ({downloaded / 1e6:.1f} MB)")

        if temp_path.stat().st_size == 0:
            raise PoseModelError(
                f"The download of {spec.display_name} produced an empty file. "
                "Check your internet connection and try again."
            )

        digest = sha256_of(temp_path)
        shutil.move(str(temp_path), str(destination))
    except urllib.error.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise PoseModelError(
            f"Could not download {spec.display_name}: the server returned "
            f"HTTP {exc.code}. The model URL may have moved."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        temp_path.unlink(missing_ok=True)
        raise PoseModelError(
            f"Could not reach {spec.source} to download {spec.display_name}. "
            "This is the only step that needs internet access; check your "
            f"connection and try again.\n\n{exc}"
        ) from exc
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise PoseModelError(
            f"Could not write the model to {destination.parent}: {exc}"
        ) from exc

    manifest = read_manifest(models_dir)
    manifest[spec.key] = {
        "filename": spec.filename,
        "sha256": digest,
        "size_bytes": destination.stat().st_size,
        "source": spec.source,
        "url": spec.url,
        "model_version": spec.model_version,
        "license": spec.license_name,
        "license_url": spec.license_url,
        "integrity": "trust-on-first-use",
    }
    _write_manifest(manifest, models_dir)

    report(1.0, f"{spec.display_name} ready")
    logger.info("Pose model %s downloaded (sha256 %s)", spec.key, digest[:16])
    return destination


def verify_model(
    spec: PoseModelSpec, models_dir: Optional[Path] = None
) -> Optional[str]:
    """Check a downloaded model against its recorded hash.

    Returns ``None`` when the file is fine, or a user-facing explanation of
    what is wrong. Returning a message rather than raising lets the UI show a
    warning next to a still-usable app.
    """
    path = model_path(spec, models_dir)
    if not path.exists():
        return f"{spec.display_name} has not been downloaded yet."

    entry = manifest_entry(spec, models_dir)
    if not entry or not entry.get("sha256"):
        return (
            f"{spec.display_name} is present but has no recorded checksum, so "
            "its integrity cannot be confirmed. Re-download it to record one."
        )

    actual = sha256_of(path)
    if actual != entry["sha256"]:
        return (
            f"{spec.display_name} does not match its recorded checksum. The "
            "file is probably truncated or corrupt. Re-download it before "
            "trusting any pose results produced with it."
        )
    return None


def ensure_model(
    spec: PoseModelSpec,
    models_dir: Optional[Path] = None,
    allow_download: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """Return a usable model path, downloading only if explicitly allowed.

    ``allow_download`` defaults to False on purpose: inference should fail with
    a clear "download this first" message rather than quietly starting a
    network transfer the user never asked for.
    """
    path = model_path(spec, models_dir)
    if path.exists() and path.stat().st_size > 0:
        return path
    if not allow_download:
        raise PoseModelError(
            f"The pose model '{spec.display_name}' is not downloaded yet. "
            "Open Swing Analysis and use the Download button — it is a "
            f"one-time {spec.approx_megabytes:.0f} MB download, after which "
            "everything runs offline."
        )
    return download_model(spec, models_dir, progress)
