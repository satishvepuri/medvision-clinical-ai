@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m mlflow ui --host 127.0.0.1 --port 5000
