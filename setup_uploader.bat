@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHON_COMMAND="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined PYTHON_COMMAND (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PYTHON_COMMAND=py -%%V"
    )
)

if not defined PYTHON_COMMAND (
    python -c "import sys; major, minor = sys.version_info[:2]; raise SystemExit(0 if major == 3 and 10 <= minor <= 13 else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_COMMAND=python"
)
if not defined PYTHON_COMMAND goto no_python

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating lightweight uploader environment...
    %PYTHON_COMMAND% -m venv .venv
    if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\python.exe" exit /b 1

echo [2/2] Checking lightweight uploader dependency...
".venv\Scripts\python.exe" -c "import supabase" >nul 2>nul
if errorlevel 1 (
    echo Installing Supabase uploader dependency...
    ".venv\Scripts\python.exe" -m pip install -r local_uploader\requirements-uploader.txt
    if errorlevel 1 exit /b 1
)

echo Listener setup complete. Returning to the folder listener...
exit /b 0

:no_python
echo Python 3.10, 3.11, 3.12, or 3.13 was not found.
echo Install one of these Python versions, then run this file again.
exit /b 1
