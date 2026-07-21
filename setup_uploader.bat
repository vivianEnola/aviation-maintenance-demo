@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto no_python

set "PYTHON_VERSION="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYTHON_VERSION (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PYTHON_VERSION=%%V"
    )
)
if not defined PYTHON_VERSION goto no_python

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating lightweight uploader environment...
    py -%PYTHON_VERSION% -m venv .venv
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
