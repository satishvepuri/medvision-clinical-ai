@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\check_environment.py
".venv\Scripts\python.exe" -m pytest -q
