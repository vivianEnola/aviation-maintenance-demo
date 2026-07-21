@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto no_python

py -3.12 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 goto no_python

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating lightweight uploader environment...
    py -3.12 -m venv .venv
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
echo Python 3.12 was not found. Install Python 3.12, then run this file again.
exit /b 1
