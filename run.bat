@echo off
title Automatic Video Highlight Detection - Flask Server
echo ======================================================================
echo Starting HighlightAI Web Server...
echo URL:   http://127.0.0.1:5000
echo ======================================================================

set "PYTHON_EXE=python"
if exist "C:\ProgramData\anaconda3\python.exe" set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

"%PYTHON_EXE%" app.py
pause
