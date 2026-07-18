# QuantJourney Backtester on Windows

This guide uses native Windows tools. WSL and Git Bash are not required.

## Requirements

- Windows 10 or 11
- Git for Windows
- Python 3.11, 3.12, 3.13 or 3.14 from python.org

During Python installation, enable **Add Python to PATH** and install the
Python launcher (`py`).

## Install from GitHub

Open PowerShell or Command Prompt:

```powershell
git clone https://github.com/QuantJourneyOrg/quantjourney-bt.git
cd quantjourney-bt

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -e ".[data,reports]"
```

The launcher uses `.venv\Scripts\python.exe` directly, so activating the
environment is optional.

## Run the deterministic sample

PowerShell:

```powershell
.\strategy.bat --list
.\strategy.bat example_weights_01_sma_daily --check
.\strategy.bat example_weights_01_sma_daily --sample-data --output .\reports
```

Command Prompt:

```bat
strategy.bat --list
strategy.bat example_weights_01_sma_daily --check
strategy.bat example_weights_01_sma_daily --sample-data --output reports
```

Open the local report from PowerShell:

```powershell
Start-Process .\reports\ExampleWeights01_DailySMATrend\dashboard.html
```

## Use real market data

Create `.env` in the repository root. It is ignored by Git:

```dotenv
QJ_API_KEY=QJ_live_your_key_here
```

`strategy.bat` loads this file automatically. Never commit `.env`, paste the
key into a strategy, or show it in terminal output.

Run the same strategy without `--sample-data`:

```powershell
.\strategy.bat example_weights_01_sma_daily --output .\reports
```

You can also set credentials only for the current PowerShell session:

```powershell
$env:QJ_API_KEY = "QJ_live_your_key_here"
.\strategy.bat example_weights_01_sma_daily
```

## Useful commands

```powershell
.\strategy.bat --help
.\strategy.bat --all --check
.\strategy.bat example_weights_01_sma_daily --quiet
.\strategy.bat example_weights_01_sma_daily --no-reports
.\strategy.bat example_weights_01_sma_daily --output .\reports\demo
```

## Troubleshooting

### `py` is not recognized

Reinstall Python from python.org with the Python launcher enabled, or replace
`py -3.11` with the full path to `python.exe`.

### PowerShell blocks `Activate.ps1`

Activation is not required. Use `.\.venv\Scripts\python.exe` and
`.\strategy.bat` exactly as shown above.

### Virtual environment not found

Run this command from the repository root:

```powershell
py -3.11 -m venv .venv
```

### Missing dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
```

### No credentials set

Use `--sample-data` for the offline demo, or add `QJ_API_KEY` to `.env` for the
authenticated real-data path.
