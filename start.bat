@echo off
REM ShufenTools Startup Script
REM This script will set up the virtual environment and start the Flask app

echo ========================================
echo ShufenTools - People Icon Remover
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [Step 1/3] Creating virtual environment...
    py -3.11 -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create virtual environment.
        echo Please make sure Python 3.11 is installed.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo [Step 1/3] Virtual environment found.
    echo.
)

REM Activate virtual environment
echo [Step 2/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
pip show flask >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to install dependencies.
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
    echo.
) else (
    echo Dependencies already installed.
    echo.
)

REM Start the Flask app
echo [Step 3/3] Starting ShufenTools...
echo.
echo ========================================
echo Server will start at: http://localhost:5001
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

pause