# Public release process

Public releases are built only by GitHub Actions from a clean `vX.Y.Z` tag on
`main`. Local uploads are not supported.

This checkout uses the tracked `.githooks/` guards. An untracked
`RELEASE_BLOCKED` rejects local commits and pushes until the marker is staged
with the reviewed hardening commit. Once tracked, ordinary review commits may
be pushed, but the PyPI workflow remains fail-closed until a dedicated release
commit removes the marker.

Enable the tracked hooks once in every fresh clone:

```bash
git config core.hooksPath .githooks
```

## Release checklist

1. Confirm all required CI jobs pass on `main`.
2. Review every change to `release/public_artifacts.txt`. This file is the
   approved wheel/sdist boundary; undeclared runtime, strategy, and
   distributable documentation files fail CI.
3. Review `uv.lock` and any change to `quality/mypy-baseline.json`; the
   baseline must reflect a deliberate diagnostic review, not a CI bypass.
4. Update the version in both `pyproject.toml` and the fallback in
   `backtester/version.py`, then add the matching release section to
   `CHANGELOG.md`. Update the explicit "Current PyPI release" line in
   `README.md` after the version is confirmed on PyPI.
5. When repository launch behavior changes, keep `strategy.py`, `strategy.sh`,
   `strategy.bat`, `WINDOWS.md`, README examples, Windows CI, and the
   Windows/macOS/Linux paths on `/how-to-start` synchronized.
6. Remove `RELEASE_BLOCKED` only in the dedicated reviewed release commit.
7. Tag that exact commit as `vX.Y.Z` and push the tag. The publish workflow
   verifies that the tag is on `main`, the checkout is clean, tests and the
   full lint/format/type gates pass, and wheel/sdist payloads exactly match the
   approved manifest before trusted publishing is allowed.

If a build fails, fix it in a new commit and create a new version/tag. Do not
move or reuse a published tag, and never upload artifacts produced from a local
working tree.
