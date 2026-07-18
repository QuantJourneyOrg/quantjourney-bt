#!/usr/bin/env python3
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.
"""Fail-closed source, version, and distribution checks for public releases."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release" / "public_artifacts.txt"
TOP_LEVEL_SDIST = {
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "WINDOWS.md",
    "compare/README.md",
    "pyproject.toml",
    "strategy.bat",
    "strategy.py",
    "strategy.sh",
}
SDIST_ROOT_SUFFIXES = {
    "backtester": {".py", ".typed"},
    "benchmarks": {".md"},
    "docs": {".md", ".mdx"},
    "skills": {".md"},
    "strategies": {".md", ".py"},
}
FORBIDDEN_PUBLIC_PATHS = {
    "backtester/engines/archive.py",
    "backtester/engines/factsheet_pdf.py",
    "backtester/engines/narrative.py",
    "backtester/engines/pdf_creation.py",
    "backtester/engines/plot_orchestrator.py",
    "backtester/portfolio/blotter_plots.py",
    "backtester/portfolio/crisis_analysis.py",
    "backtester/portfolio/portfolio_plots_extra.py",
    "backtester/portfolio/strategy_trace_plots.py",
    "backtester/utils/reproducibility.py",
}
FORBIDDEN_METADATA_TOKENS = {
    "_repo_qj_backtester_private",
    "_repo_qj_backtester_web",
    "ADR-113",
    "ADR-114",
    "Restored the private package",
}


def _run_git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _load_manifest(path: Path) -> tuple[set[str], set[str]]:
    wheel: set[str] = set()
    sdist: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            target, relative_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: expected '<target> <path>'") from exc
        if target not in {"both", "sdist", "wheel"}:
            raise ValueError(f"{path}:{line_number}: invalid target {target!r}")
        normalized = PurePosixPath(relative_path).as_posix()
        if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
            raise ValueError(f"{path}:{line_number}: unsafe path {relative_path!r}")
        if target in {"both", "wheel"}:
            wheel.add(normalized)
        if target in {"both", "sdist"}:
            sdist.add(normalized)
    return wheel, sdist


def _is_sdist_source(path: str) -> bool:
    pure = PurePosixPath(path)
    if path in TOP_LEVEL_SDIST:
        return True
    if not pure.parts:
        return False
    suffixes = SDIST_ROOT_SUFFIXES.get(pure.parts[0])
    return suffixes is not None and pure.suffix in suffixes


def _source_candidates() -> tuple[set[str], set[str]]:
    wheel: set[str] = set()
    sdist: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(
            part in {".git", ".venv", "__pycache__", "build", "dist"} for part in relative.parts
        ):
            continue
        normalized = relative.as_posix()
        if (
            relative.parts
            and relative.parts[0].startswith("backtester")
            and path.suffix in {".py", ".typed"}
        ):
            wheel.add(normalized)
        if _is_sdist_source(normalized):
            sdist.add(normalized)
    return wheel, sdist


def _assert_equal(label: str, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = [f"{label} does not match the approved public manifest."]
        if missing:
            details.append("Missing:\n  " + "\n  ".join(missing))
        if unexpected:
            details.append("Unexpected:\n  " + "\n  ".join(unexpected))
        raise RuntimeError("\n".join(details))


def verify_source_manifest(wheel_expected: set[str], sdist_expected: set[str]) -> None:
    wheel_actual, sdist_actual = _source_candidates()
    _assert_equal("Wheel source", wheel_actual, wheel_expected)
    _assert_equal("Sdist source", sdist_actual, sdist_expected)
    forbidden = sorted((wheel_actual | sdist_actual) & FORBIDDEN_PUBLIC_PATHS)
    if forbidden:
        raise RuntimeError("Forbidden private paths found:\n  " + "\n  ".join(forbidden))


def verify_clean_tree() -> None:
    status = _run_git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("Release requires a clean checkout; git status is not empty:\n" + status)


def _project_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _distribution_stem() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    normalized_name = re.sub(r"[-_.]+", "_", str(data["project"]["name"]))
    return f"{normalized_name}-{data['project']['version']}"


def _project_scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict) or not all(
        isinstance(name, str) and isinstance(target, str) for name, target in scripts.items()
    ):
        raise RuntimeError("pyproject.toml [project.scripts] must map names to import targets")
    return dict(sorted(scripts.items()))


def _expected_entry_points_text() -> str:
    scripts = _project_scripts()
    if not scripts:
        return ""
    return "\n".join(
        ["[console_scripts]", *(f"{name} = {target}" for name, target in scripts.items())]
    )


def verify_version_metadata() -> str:
    """Require one consistent package version across source and changelog."""
    version = _project_version()
    version_source = (ROOT / "backtester" / "version.py").read_text(encoding="utf-8")
    fallback = re.search(r'return\s+["\']([^"\']+)["\']', version_source)
    if fallback is None or fallback.group(1) != version:
        raise RuntimeError("pyproject.toml and backtester/version.py fallback versions differ")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {version} " not in changelog and f"## {version}\n" not in changelog:
        raise RuntimeError(f"CHANGELOG.md has no release section for {version}")
    release_headings = re.findall(r"^## (.+)$", changelog, flags=re.MULTILINE)
    if not release_headings or not release_headings[0].startswith(version):
        raise RuntimeError(
            "The first CHANGELOG.md release section must be the package version; "
            "move Unreleased notes into the versioned section before tagging"
        )
    return version


def verify_version_and_tag(trigger_tag: str | None = None) -> None:
    version = verify_version_metadata()
    expected_tag = f"v{version}"
    if trigger_tag is not None and trigger_tag != expected_tag:
        raise RuntimeError(
            f"Workflow tag {trigger_tag!r} does not match package tag {expected_tag!r}"
        )
    tags = set(_run_git("tag", "--points-at", "HEAD").splitlines())
    if expected_tag not in tags:
        raise RuntimeError(f"HEAD must be tagged exactly with {expected_tag}")


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive path: {name}")
    return path


def verify_wheel(path: Path, expected: set[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        raw_names = archive.namelist()
        if len(raw_names) != len(set(raw_names)):
            raise RuntimeError("Wheel contains duplicate archive members")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Wheel CRC check failed for {corrupt_member}")
        names = {_safe_archive_path(name).as_posix() for name in raw_names}
        dist_info_roots = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        if len(dist_info_roots) != 1:
            raise RuntimeError("Wheel must contain exactly one dist-info directory")
        dist_info_root = next(iter(dist_info_roots))
        expected_dist_info = f"{_distribution_stem()}.dist-info"
        if dist_info_root != expected_dist_info:
            raise RuntimeError(
                f"Wheel dist-info {dist_info_root!r} does not match {expected_dist_info!r}"
            )
        generated = {
            f"{dist_info_root}/METADATA",
            f"{dist_info_root}/RECORD",
            f"{dist_info_root}/WHEEL",
            f"{dist_info_root}/licenses/LICENSE",
            f"{dist_info_root}/top_level.txt",
        }
        entry_points_path = f"{dist_info_root}/entry_points.txt"
        if _project_scripts():
            generated.add(entry_points_path)
        _assert_equal(
            "Wheel generated metadata",
            {name for name in names if name.startswith(f"{dist_info_root}/")},
            generated,
        )
        payload = names - generated
        _assert_equal("Wheel payload", payload, expected)
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise RuntimeError("Wheel must contain exactly one METADATA file")
        metadata = archive.read(metadata_files[0]).decode("utf-8", errors="replace")
        offenders = sorted(token for token in FORBIDDEN_METADATA_TOKENS if token in metadata)
        if offenders:
            raise RuntimeError("Wheel metadata exposes private tokens: " + ", ".join(offenders))
        if entry_points_path in generated:
            actual_entry_points = archive.read(entry_points_path).decode("utf-8").strip()
            if actual_entry_points != _expected_entry_points_text():
                raise RuntimeError("Wheel console entry points do not match pyproject.toml")


def verify_sdist(path: Path, expected: set[str]) -> None:
    with tarfile.open(path, mode="r:*") as archive:
        members = archive.getmembers()
        if any(member.issym() or member.islnk() for member in members):
            raise RuntimeError("Sdist must not contain symbolic or hard links")
        root_names = {
            _safe_archive_path(member.name).parts[0]
            for member in members
            if _safe_archive_path(member.name).parts
        }
        if len(root_names) != 1:
            raise RuntimeError("Sdist must contain exactly one top-level directory")
        distribution_root = next(iter(root_names))
        expected_root = _distribution_stem()
        if distribution_root != expected_root:
            raise RuntimeError(f"Sdist root {distribution_root!r} does not match {expected_root!r}")
        distribution_name = distribution_root.rsplit("-", 1)[0]
        generated = {
            "PKG-INFO",
            "setup.cfg",
            f"{distribution_name}.egg-info/PKG-INFO",
            f"{distribution_name}.egg-info/SOURCES.txt",
            f"{distribution_name}.egg-info/dependency_links.txt",
            f"{distribution_name}.egg-info/requires.txt",
            f"{distribution_name}.egg-info/top_level.txt",
        }
        if _project_scripts():
            generated.add(f"{distribution_name}.egg-info/entry_points.txt")
        files: set[str] = set()
        for member in members:
            if not member.isfile():
                continue
            archive_path = _safe_archive_path(member.name)
            if len(archive_path.parts) < 2 or archive_path.parts[0] != distribution_root:
                raise RuntimeError(f"File outside sdist root: {member.name}")
            normalized = PurePosixPath(*archive_path.parts[1:]).as_posix()
            files.add(normalized)
        _assert_equal("Sdist payload", files, expected | generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--tag", help="Tag that triggered the release workflow")
    parser.add_argument("--require-unblocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheel_expected, sdist_expected = _load_manifest(args.manifest)
    verify_source_manifest(wheel_expected, sdist_expected)
    verify_version_metadata()
    if args.require_clean:
        verify_clean_tree()
    if args.require_tag:
        verify_version_and_tag(args.tag)
    if args.require_unblocked and (ROOT / "RELEASE_BLOCKED").exists():
        raise RuntimeError("Public release is intentionally blocked by RELEASE_BLOCKED")
    if args.wheel:
        verify_wheel(args.wheel, wheel_expected)
    if args.sdist:
        verify_sdist(args.sdist, sdist_expected)
    print("Public release verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
