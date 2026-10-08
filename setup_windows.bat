@echo off
setlocal
cd /d "%~dp0"
echo.
echo [1/3] Creating Python virtual environment...
py -3.10 -m venv .venv
if errorlevel 1 (
  echo Could not create the environment. Make sure Python 3.10 is installed.
  exit /b 1
)
echo.
echo [2/3] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo.
echo [3/3] Installing project packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo Setup complete.
echo Next run: create_demo_data.bat
endlocal
