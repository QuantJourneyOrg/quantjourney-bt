# Contributing

Thanks for your interest in the QuantJourney Backtester. Contributions are
welcome — especially **new example strategies**, bug fixes, and documentation
improvements.

## How contributions are accepted

**Open a pull request on GitHub.** A maintainer reviews every pull request and
decides how to integrate it. A change may be merged as-is, adapted first, or
re-applied through the QuantJourney source of truth and the pull request closed
as *integrated*. Either way you will get a response, and accepted work is
credited. Please don't be surprised if your change lands in a slightly different
form — the maintainer curates how each contribution is included.

Small, focused pull requests are reviewed fastest: one strategy, or one fix, per
pull request.

## Ways to contribute

- **New example strategies** — the easiest and most valued contribution. A good
  example is small, self-contained, and teaches one idea clearly.
- **Bug fixes** — with a short description of the incorrect behavior and, where
  possible, a test that fails before the fix.
- **Documentation** — clarifications, typos, better explanations.
- **Ideas and feedback** — open an issue; not every contribution needs code.

## Step by step

### 1. Fork the repository

On GitHub, click **Fork** (top right) to create your own copy.

### 2. Clone your fork and add the upstream remote

```bash
git clone https://github.com/<your-username>/quantjourney-bt.git
cd quantjourney-bt
git remote add upstream https://github.com/QuantJourneyOrg/quantjourney-bt.git
```

### 3. Sync your main branch with upstream

```bash
git checkout main
git pull upstream main
```

### 4. Create a feature branch

```bash
git checkout -b feat/short-description
```

Use a short, descriptive name, e.g. `feat/example-bollinger-squeeze` or
`fix/limit-fill-rounding`.

### 5. Set up a development environment

Use the reviewed lockfile (do not install into system/Homebrew Python):

```bash
uv sync --frozen --extra dev --extra data --extra wf --extra typecheck
```

### 6. Make your change

For a new strategy, follow the conventions in
[Adding an example strategy](#adding-an-example-strategy) below.

### 7. Run the checks locally

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python tools/check_mypy_baseline.py
# for a new strategy, confirm it imports cleanly:
./strategy.sh <your_strategy_name> --check
```

Everything should pass before you open a pull request.
Do not update `quality/mypy-baseline.json` merely to make CI green; a
maintainer must review the complete diagnostic change.

### 8. Commit

```bash
git add .
git commit -m "Add Bollinger squeeze example strategy"
```

Write a clear, present-tense message describing what the change does.

### 9. Push to your fork

```bash
git push origin feat/short-description
```

### 10. Open the pull request

Go to your fork on GitHub — it will offer to **Compare & pull request** against
`QuantJourneyOrg/quantjourney-bt`. Open it, and in the description include:

- what the change does and why,
- for a strategy: the idea in one or two sentences and the universe used,
- confirmation that `pytest`, `ruff`, and (for strategies) `--check` pass.

### 11. Respond to review

A maintainer will review and may ask for adjustments or adapt the change during
integration. Push more commits to the same branch to update the pull request.

## Adding an example strategy

Example strategies follow a simple convention so they stay consistent and
discoverable:

- **File name:** `example_<mode>_<NN>_<name>.py`, where `<mode>` is `weights`,
  `orders`, or `wf` (walk-forward / optimization), and `<NN>` is the next number
  in that series.
- **License header + docstring:** start with the short license header used by the
  other files, then a docstring describing Mode, the idea, the universe, and
  (for order or intraday strategies) the order type / granularity — match the
  style of the existing files in `strategies/`.
- **Structure:** subclass `Backtester` and implement `_compute_signals` and
  `_compute_weights` (weight mode) or `_compute_orders` (order mode).
- **Keep it runnable:** it should pass `./strategy.sh <name> --check` (an
  import-only check, no credentials or data call).
- **Add it to the catalog:** a one-line row in
  [`strategies/README.md`](strategies/README.md).

Prefer widely available symbols so the example runs against common data, and
keep the universe small enough to read the resulting report.

## Pull request checklist

- [ ] `pytest -q` passes.
- [ ] `ruff check .` is clean.
- [ ] `ruff format --check .` and the reviewed mypy baseline gate pass.
- [ ] New strategies pass `./strategy.sh <name> --check`.
- [ ] Docstrings follow the existing style.
- [ ] No credentials, API keys, tokens, or private paths are included.
- [ ] The change is focused and described in the pull request.

## Scope and honesty

Example strategies are research templates, not production trading systems. If an
example simplifies an assumption (borrow cost, financing, liquidity, market
impact), state it in the docstring so the assumption is documented, not hidden —
this is the core value of the project.

## Code of conduct

Be respectful and constructive. Assume good faith, keep discussion technical,
and help newcomers. Behavior that harasses or demeans others is not welcome.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).
