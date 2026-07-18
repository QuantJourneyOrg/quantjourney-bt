@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" if exist "%ROOT%..\.venv\Scripts\python.exe" set "PYTHON=%ROOT%..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Error: Virtual environment not found.
    echo Run: py -3.11 -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -e ".[dev,data]"
    exit /b 1
)

"%PYTHON%" "%ROOT%strategy.py" %*
exit /b %ERRORLEVEL%
