@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto no_python

py -3.12 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 goto no_python

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating Python 3.12 environment...
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)

echo [2/2] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Setup complete. Run run_app.bat to open the interface.
exit /b 0

:no_python
echo Python 3.12 was not found.
echo Install the 64-bit Python 3.12 release from https://www.python.org/downloads/
echo Enable "Add python.exe to PATH", then run this file again.
exit /b 1
