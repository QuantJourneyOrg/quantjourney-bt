# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Regression tests for fail-closed public artifact verification."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_public_release", ROOT / "tools" / "verify_public_release.py"
)
assert SPEC is not None and SPEC.loader is not None
release_verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_verifier)


def _approved_payloads() -> tuple[set[str], set[str]]:
    return release_verifier._load_manifest(ROOT / "release" / "public_artifacts.txt")


def _write_wheel(
    path: Path,
    payload: set[str],
    *,
    unexpected: str | None = None,
    entry_points: bytes | None = None,
) -> None:
    dist_info = f"{release_verifier._distribution_stem()}.dist-info"
    entry_points = entry_points or release_verifier._expected_entry_points_text().encode()
    generated = {
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\nName: quantjourney-bt\n",
        f"{dist_info}/RECORD": b"",
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\n",
        f"{dist_info}/entry_points.txt": entry_points,
        f"{dist_info}/licenses/LICENSE": b"Apache-2.0\n",
        f"{dist_info}/top_level.txt": b"backtester\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(payload):
            archive.writestr(name, b"")
        for name, content in generated.items():
            archive.writestr(name, content)
        if unexpected:
            archive.writestr(unexpected, b"private content")


def _write_sdist(path: Path, payload: set[str], *, unexpected: str | None = None) -> None:
    root = release_verifier._distribution_stem()
    generated = {
        "PKG-INFO",
        "setup.cfg",
        "quantjourney_bt.egg-info/PKG-INFO",
        "quantjourney_bt.egg-info/SOURCES.txt",
        "quantjourney_bt.egg-info/dependency_links.txt",
        "quantjourney_bt.egg-info/entry_points.txt",
        "quantjourney_bt.egg-info/requires.txt",
        "quantjourney_bt.egg-info/top_level.txt",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(payload | generated | ({unexpected} if unexpected else set())):
            content = b""
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_wheel_rejects_unapproved_top_level_package(tmp_path: Path) -> None:
    wheel_payload, _ = _approved_payloads()
    wheel = tmp_path / "package.whl"
    _write_wheel(wheel, wheel_payload, unexpected="backtester_private/leak.py")

    with pytest.raises(RuntimeError, match="Wheel payload"):
        release_verifier.verify_wheel(wheel, wheel_payload)


def test_sdist_rejects_unapproved_test_or_private_file(tmp_path: Path) -> None:
    _, sdist_payload = _approved_payloads()
    sdist = tmp_path / "package.tar.gz"
    _write_sdist(sdist, sdist_payload, unexpected="tests/private_contract.py")

    with pytest.raises(RuntimeError, match="Sdist payload"):
        release_verifier.verify_sdist(sdist, sdist_payload)


def test_wheel_rejects_modified_console_entry_point(tmp_path: Path) -> None:
    wheel_payload, _ = _approved_payloads()
    wheel = tmp_path / "package.whl"
    _write_wheel(
        wheel,
        wheel_payload,
        entry_points=b"[console_scripts]\nqj-bt = unreviewed.module:main\n",
    )

    with pytest.raises(RuntimeError, match="console entry points"):
        release_verifier.verify_wheel(wheel, wheel_payload)
