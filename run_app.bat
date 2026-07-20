@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup_windows.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

echo Opening the app at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
