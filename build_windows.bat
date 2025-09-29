@echo off
REM Build a Windows executable for the ShufenTools Flask application.
REM Requirements:
REM   - Python 3.11+ installed and on PATH
REM   - pip install -r requirements.txt
REM   - pip install pyinstaller

set APP_NAME=shufen_tools
set SPEC_FILE=%APP_NAME%.spec

if exist dist rd /s /q dist
if exist build rd /s /q build
if exist %SPEC_FILE% del %SPEC_FILE%

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name ShufenTools ^
  --add-data "templates;templates" ^
  app.py

if %ERRORLEVEL% NEQ 0 (
  echo Build failed.
  exit /b 1
)

echo Build complete. The executable is located at dist\ShufenTools.exe
