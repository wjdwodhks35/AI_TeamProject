@echo off
rem Start the API and UI on http://localhost:8010 (local demo only).
rem Everything below is ASCII on purpose: changing the console code page
rem inside a batch file breaks label lookup on some systems.
rem   SKIP_INSTALL=1  skip dependency install
rem   RELOAD=1        restart when python code changes
setlocal
cd /d "%~dp0"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" echo Creating virtual environment (.venv) ...
if not exist "%VENV_PY%" py -3 -m venv .venv 2>nul
if not exist "%VENV_PY%" python -m venv .venv 2>nul

if not exist "%VENV_PY%" (
  echo.
  echo [ERROR] Python 3.11+ not found.
  echo         Install it from https://www.python.org/downloads/ and tick
  echo         "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)

"%VENV_PY%" "%CD%\scripts\run_local.py"
if errorlevel 1 pause
