@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup_uploader.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

if not exist "local_uploader\uploader.toml" (
    copy /Y "local_uploader\uploader.example.toml" "local_uploader\uploader.toml" >nul
)

if "%SUPABASE_URL%"=="" (
    set /p SUPABASE_URL=Enter the Supabase project URL: 
)
if "%SUPABASE_SERVICE_ROLE_KEY%"=="" (
    set /p SUPABASE_SERVICE_ROLE_KEY=Enter the Supabase uploader key: 
)
if "%SUPABASE_URL%"=="" echo Supabase project URL is required.& pause & exit /b 1
if "%SUPABASE_SERVICE_ROLE_KEY%"=="" echo Supabase uploader key is required.& pause & exit /b 1

set /p WATCH_FOLDER=Enter the local image folder path:
if "%WATCH_FOLDER%"=="" exit /b 1
set /p DEVICE_ID=Enter a device ID [mmsstv-windows-01]:
if "%DEVICE_ID%"=="" set "DEVICE_ID=mmsstv-windows-01"

".venv\Scripts\python.exe" local_uploader\watch_folder.py --watch-folder "%WATCH_FOLDER%" --device-id "%DEVICE_ID%"
pause
