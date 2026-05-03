@echo off
echo ========================================
echo Starting Dementia Backend API
echo ========================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo Starting API server on http://localhost:8000
echo Backend will now load models and start...
echo This may take 10-30 seconds.
echo.
echo Press CTRL+C to stop
echo.

python run_api.py

pause
