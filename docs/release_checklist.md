# Release Checklist

Before publishing a public release:

1. Confirm no credentials, private URLs, tokens, deployment scripts, or customer data are present.
2. Run `pytest`.
3. Run `python3 -m build`.
4. Inspect package contents with `tar -tf dist/*.tar.gz`.
5. Tag Git with `vX.Y.Z`.
6. Publish to PyPI as `quantjourney-bt`.
7. Link strategy files from the Compare page.
